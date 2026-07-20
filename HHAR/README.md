<div align="center">
  <img width="1200" height="475" alt="GHBanner" src="https://ai.google.dev/static/site-assets/images/share-ais-513315318.png" />
  <h1>HHAR - Heterogeneity Human Activity Recognition Classifier</h1>
  <p><strong>A comprehensive, interactive web environment for studying Machine Learning model performance across diverse sensory devices.</strong></p>
</div>

<br />

## 📖 Overview

The **Heterogeneity Human Activity Recognition (HHAR) Classifier** is an advanced academic web application designed to help researchers, students, and engineers understand the complexities of training machine learning models on sensor data (like accelerometers and gyroscopes). Specifically, this project focuses on **device heterogeneity** — what happens when an ML model is trained on data from one device (e.g., a Samsung smartwatch) and tested on another (e.g., an Apple smartphone).

This platform acts as a visual, interactive laboratory. Users can configure neural network hyperparameters, run synthetic training, simulate real-time sensor data, analyze statistical shifts between devices, and receive expert-level scientific analysis from an integrated AI Consultant (powered by Llama 3.3 70B via the Groq API).

---

## ✨ Core Modules

The application is divided into four main interactive modules:

### 1. Model Trainer
A visual interface allowing users to "train" a neural network on a synthetic dataset designed to model the exact physical parameters of the HHAR dataset.
- **Hyperparameter Tuning**: Users can manually adjust the Learning Rate, Batch Size, Epochs, and Hidden Layer structures.
- **Live Metrics**: Watch the training loss, validation loss, and accuracy metrics converge in real-time.
- **Confusion Matrix**: A detailed breakdown of classification accuracy across 6 specific activities: *Biking, Sitting, Standing, Walking, Stair Up, and Stair Down*.

### 2. Sensor Simulator
A real-time data visualization tool that simulates live physical movement.
- **Device Selection**: Choose between various simulated hardware devices (e.g., Nexus 4, Samsung Galaxy S3, Samsung Gear smartwatch).
- **Activity Selection**: Select an activity to simulate (e.g., Biking, Walking).
- **Data Streams**: Watch live scrolling line charts charting 3-axis Accelerometer (m/s²) and Gyroscope (rad/s) data, showcasing the distinct signal morphologies for different movements.

### 3. Heterogeneity Study
The core academic component of the app that explores the cross-device accuracy degradation (the "Heterogeneity Impact").
- **Baseline vs. Cross-Device**: Compare how a model performs when tested on the *same* device it was trained on versus an *unseen* device.
- **Statistical Analysis**: View detailed charts explaining the variance and standard deviations across sensor models, proving why device hardware and placement (wrist vs. pocket) heavily alter sensor data.

### 4. AI Consultant (Llama Integration)
An intelligent, context-aware AI lab assistant powered by **Llama 3.3 70B Versatile** (via Groq).
- **Scientific ML Analyst**: Submits the final neural network configuration, accuracy metrics, and confusion matrix to the AI. The AI returns a highly academic, markdown-formatted report evaluating signs of overfitting, biomechanical confusions (e.g., confusing stairs with walking), and actionable hyperparameter tweaks.
- **Wearable Motion Coach**: Takes live summary statistics from the Sensor Simulator (means and standard deviations of the axes) and provides rapid, friendly biomechanical feedback or athletic posture tips.

---

## 🛠️ Technology Stack

This project is built using a modern full-stack JavaScript/TypeScript architecture.

**Frontend:**
- **React 19 & React DOM** - UI library for building component-based interfaces.
- **Vite** - High-performance build tool and development server.
- **Tailwind CSS** - Utility-first CSS framework for rapid UI styling, emphasizing a sleek, dark-mode, terminal-like aesthetic.
- **Lucide React** - SVG icon library.
- **Framer Motion** (`motion`) - For fluid animations and transitions between UI states.

**Backend:**
- **Express.js** - A lightweight Node.js server. In development, it seamlessly serves the Vite frontend via middleware. In production, it statically serves the bundled frontend.
- **Groq SDK** - Official SDK used to communicate securely with the Groq inference engine to generate AI responses using the `llama-3.3-70b-versatile` model.
- **ESBuild & TSX** - Used to compile and bundle the backend TypeScript server efficiently.

---

## 🚀 Getting Started

Follow these steps to set up the project on your local machine.

### Prerequisites
- **Node.js** (v18 or higher recommended)
- **A Groq API Key** (You can get a free one from the [Groq Console](https://console.groq.com/keys))

### Installation

1. **Clone or Download the Repository:**
   Navigate into the project directory:
   ```bash
   cd HHAR
   ```

2. **Install Dependencies:**
   Install all necessary Node modules for both the frontend and backend.
   ```bash
   npm install
   ```

3. **Configure Environment Variables:**
   Copy the provided example environment file to create your own configuration.
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file in a code editor and paste your Groq API key:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   ```

### Running the Application

**Development Mode:**
To run the server and the frontend concurrently with Hot Module Replacement (HMR):
```bash
npm run dev
```
Navigate to `http://localhost:3000` in your browser. The Express server acts as a proxy, securely executing AI calls on the backend so your API key is never exposed to the client.

**Production Build:**
To build the application for deployment:
```bash
npm run build
```
This script uses Vite to bundle the React frontend into a `dist` folder, and ESBuild to bundle the Express backend into `dist/server.cjs`.

To start the production bundle:
```bash
npm run start
```

---

## 🌐 Security & Architecture Note

To ensure API security, this application follows the **Backend-for-Frontend (BFF)** pattern. The frontend never communicates directly with the Groq API. Instead, when a user requests an AI analysis, the React frontend sends a POST request to our local Express server endpoints (`/api/analyze-model` or `/api/analyze-stream`). The Express server attaches the `GROQ_API_KEY`, securely queries the Llama model, and returns the response to the frontend. 

---
<div align="center">
  <p><i>Developed as an educational tool for UCI HAR Laboratory standards • v0.9 Beta</i></p>
</div>
