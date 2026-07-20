import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";
import Groq from "groq-sdk";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Initialize Groq client server-side
  const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY
  });

  // Helper to call Groq with robust exponential backoff retry for transient errors (e.g. 503, 429)
  async function callGroqWithRetry(params: { model: string; contents: string }, maxRetries = 4) {
    let attempt = 0;
    let delay = 1000;
    while (attempt < maxRetries) {
      try {
        const chatCompletion = await groq.chat.completions.create({
          messages: [{ role: "user", content: params.contents }],
          model: params.model,
        });
        return { text: chatCompletion.choices[0]?.message?.content || "" };
      } catch (error: any) {
        attempt++;
        const errorMessage = error?.message || String(error);
        const errorStatus = error?.status || error?.statusCode || error?.code;
        const isTransient = errorMessage.includes("503") || 
                            errorMessage.includes("UNAVAILABLE") || 
                            errorMessage.includes("rate limit") ||
                            errorStatus === 503 ||
                            errorStatus === 429;
        
        if (isTransient && attempt < maxRetries) {
          console.warn(`Transient Groq error encountered (attempt ${attempt}/${maxRetries}): ${errorMessage}. Retrying in ${delay}ms...`);
          await new Promise((resolve) => setTimeout(resolve, delay));
          delay *= 2.5; // exponential backoff
        } else {
          throw error;
        }
      }
    }
    throw new Error("Max retries exceeded");
  }

  // API Health check
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", time: new Date().toISOString() });
  });

  // Model training analysis API using Gemini 3.5 Flash
  app.post("/api/analyze-model", async (req, res) => {
    try {
      const { hyperparameters, finalMetrics, confusionMatrix, deviceSetup } = req.body;

      if (!process.env.GROQ_API_KEY) {
        return res.status(500).json({ 
          error: "GROQ_API_KEY environment variable is not configured. Please add it to your .env file." 
        });
      }

      const activities = ["Biking", "Sitting", "Standing", "Walking", "Stair Up", "Stair Down"];
      
      const formattedConfusionMatrix = confusionMatrix
        ? confusionMatrix.map((row: number[], i: number) => {
            return `${activities[i]}: [${row.join(", ")}]`;
          }).join("\n")
        : "Not provided";

      const prompt = `You are an expert biomechanics engineer and machine learning researcher specializing in wearable human activity recognition (HAR).
An interactive neural network has just finished training on a synthetic dataset designed to model the exact physical parameters of the Heterogeneity Human Activity Recognition (HHAR) dataset.

Here are the details of the training run:
- Device configuration: ${deviceSetup}
- Hyperparameters:
  * Learning Rate: ${hyperparameters.learningRate}
  * Epochs: ${hyperparameters.epochs}
  * Batch Size: ${hyperparameters.batchSize}
  * Hidden Layers: ${JSON.stringify(hyperparameters.hiddenLayers)}
- Final Training Metrics:
  * Final Training Loss: ${finalMetrics.trainLoss ? finalMetrics.trainLoss.toFixed(4) : "N/A"}
  * Final Validation Loss: ${finalMetrics.valLoss ? finalMetrics.valLoss.toFixed(4) : "N/A"}
  * Final Training Accuracy: ${finalMetrics.trainAcc ? (finalMetrics.trainAcc * 100).toFixed(2) : "N/A"}%
  * Final Validation Accuracy: ${finalMetrics.valAcc ? (finalMetrics.valAcc * 100).toFixed(2) : "N/A"}%

Confusion Matrix (Rows are True labels, Columns are Predicted labels in order of [Biking, Sitting, Standing, Walking, Stair Up, Stair Down]):
${formattedConfusionMatrix}

Please perform a structured, scientific, and highly engaging analysis covering:
1. **Model Performance Summary**: How well did the network learn? Are there signs of overfitting or underfitting given the train/val gap?
2. **Biomechanical Confusion Analysis**: Pinpoint any specific activity confusions (e.g. Walking vs Stairs, or Sitting vs Standing). Explain the biomechanical similarities in sensor data (gravity vectors, cadence, cycle variance) causing these errors.
3. **The Heterogeneity Impact**: Explain how device heterogeneity (specifically pocket-worn smartphones vs. wrist-worn smartwatches) impacts signal morphology (e.g., wrist rotates heavily during walking, while pockets translate linear acceleration). How does this affect generalization if we train on one and test on another?
4. **Actionable Hyperparameter Recommendations**: Provide 2 target recommendations on how to tweak this neural network architecture, learning rate, or features to improve classification accuracy.

Keep your tone professional, encouraging, and highly academic yet clear. Keep formatting clean with beautiful Markdown headings.`;

      const response = await callGroqWithRetry({
        model: "llama-3.3-70b-versatile",
        contents: prompt,
      });

      res.json({ analysis: response.text });
    } catch (error: any) {
      console.error("Groq analysis error:", error);
      res.status(500).json({ error: error.message || "Failed to analyze model configuration" });
    }
  });

  // Live stream activity advisor endpoint
  app.post("/api/analyze-stream", async (req, res) => {
    try {
      const { activity, device, samplesSummary } = req.body;

      if (!process.env.GROQ_API_KEY) {
        return res.status(500).json({ 
          error: "GROQ_API_KEY environment variable is not configured. Please add it to your .env file." 
        });
      }

      const prompt = `You are a biomechanical performance coach. The user is currently simulating or recording sensor data of the activity: "${activity}" using the device: "${device}".
The current summary of sensor readings (standard deviations and mean averages) is:
- Accelerometer Mean: [${samplesSummary.mean_ax.toFixed(3)}, ${samplesSummary.mean_ay.toFixed(3)}, ${samplesSummary.mean_az.toFixed(3)}] m/s²
- Accelerometer Standard Deviation: [${samplesSummary.std_ax.toFixed(3)}, ${samplesSummary.std_ay.toFixed(3)}, ${samplesSummary.std_az.toFixed(3)}] m/s²
- Gyroscope Mean: [${samplesSummary.mean_gx.toFixed(3)}, ${samplesSummary.mean_gy.toFixed(3)}, ${samplesSummary.mean_gz.toFixed(3)}] rad/s
- Gyroscope Standard Deviation: [${samplesSummary.std_gx.toFixed(3)}, ${samplesSummary.std_gy.toFixed(3)}, ${samplesSummary.std_gz.toFixed(3)}] rad/s

Based on these physical parameters:
1. Briefly explain how this device-specific sensor signature matches the physical movement profile of "${activity}".
2. Offer 1-2 interesting athletic, posture, or physical therapy suggestions related to this activity (e.g. correct pedaling form if biking, step cadence for walking, or micro-breaks for sitting).

Keep your response very short, friendly, and structured (under 150 words total).`;

      const response = await callGroqWithRetry({
        model: "llama-3.3-70b-versatile",
        contents: prompt,
      });

      res.json({ advice: response.text });
    } catch (error: any) {
      console.error("Groq stream advisor error:", error);
      res.status(500).json({ error: error.message || "Failed to analyze sensor stream" });
    }
  });

  // Integrate Vite server in development mode, otherwise serve production static files
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT} in ${process.env.NODE_ENV || "development"} mode`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
