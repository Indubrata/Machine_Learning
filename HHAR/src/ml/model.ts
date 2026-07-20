/**
 * Multi-Layer Perceptron (MLP) Neural Network from scratch in pure TypeScript.
 * Designed to train interactively in the browser/node environment.
 */

export interface ModelHyperparameters {
  learningRate: number;
  epochs: number;
  batchSize: number;
  hiddenLayers: number[]; // e.g., [32, 16]
}

export interface TrainingProgress {
  epoch: number;
  trainLoss: number;
  trainAcc: number;
  valLoss: number;
  valAcc: number;
}

export class DenseLayer {
  weights: number[][]; // [outputs][inputs]
  biases: number[];    // [outputs]
  
  // Cache for backprop
  inputs: number[] = [];
  outputs: number[] = [];
  z: number[] = []; // pre-activation
  
  // Gradients for accumulation
  dWeights: number[][] = [];
  dBiases: number[] = [];

  constructor(public inputSize: number, public outputSize: number) {
    // He (Kaiming) Initialization for weights
    const scale = Math.sqrt(2.0 / inputSize);
    
    this.weights = Array.from({ length: outputSize }, () => 
      Array.from({ length: inputSize }, () => this.gaussianRandom() * scale)
    );
    this.biases = Array(outputSize).fill(0.0);

    this.clearGradients();
  }

  private gaussianRandom(): number {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); 
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  clearGradients() {
    this.dWeights = Array.from({ length: this.outputSize }, () => Array(this.inputSize).fill(0));
    this.dBiases = Array(this.outputSize).fill(0);
  }

  forward(inputs: number[], activate: (x: number) => number): number[] {
    this.inputs = [...inputs];
    const outputs = Array(this.outputSize).fill(0);
    this.z = Array(this.outputSize).fill(0);

    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.biases[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += this.weights[i][j] * inputs[j];
      }
      this.z[i] = sum;
      outputs[i] = activate(sum);
    }

    this.outputs = outputs;
    return outputs;
  }

  // Update weights based on gradients and learning rate
  updateWeights(lr: number) {
    for (let i = 0; i < this.outputSize; i++) {
      this.biases[i] -= lr * this.dBiases[i];
      for (let j = 0; j < this.inputSize; j++) {
        this.weights[i][j] -= lr * this.dWeights[i][j];
      }
    }
  }
}

export class NeuralNetwork {
  layers: DenseLayer[] = [];
  classes: string[];

  constructor(inputSize: number, hiddenSizes: number[], outputSize: number, classes: string[]) {
    this.classes = classes;
    const sizes = [inputSize, ...hiddenSizes, outputSize];
    
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(new DenseLayer(sizes[i], sizes[i + 1]));
    }
  }

  // Activations
  private relu(x: number): number {
    return Math.max(0, x);
  }

  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  private softmax(arr: number[]): number[] {
    const maxVal = Math.max(...arr); // stability trick
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / (sumExps || 1));
  }

  // Forward Propagation
  forward(inputs: number[]): number[] {
    let currentActivations = [...inputs];
    
    for (let i = 0; i < this.layers.length; i++) {
      const isOutputLayer = (i === this.layers.length - 1);
      
      if (isOutputLayer) {
        // Output layer forward pass (linear first, then softmax outside)
        currentActivations = this.layers[i].forward(currentActivations, x => x);
        currentActivations = this.softmax(currentActivations);
        this.layers[i].outputs = [...currentActivations]; // cache softmax output
      } else {
        // Hidden layers forward pass with ReLU
        currentActivations = this.layers[i].forward(currentActivations, this.relu);
      }
    }
    
    return currentActivations;
  }

  // Backpropagation for a single training sample
  backpropagate(inputs: number[], targetLabel: number) {
    const outputs = this.forward(inputs);
    
    // Create one-hot target
    const target = Array(this.layers[this.layers.length - 1].outputSize).fill(0);
    target[targetLabel] = 1;

    // 1. Output layer error (Cross-entropy with Softmax simplifies beautifully)
    // delta_out = output_activations - target_one_hot
    const outputLayer = this.layers[this.layers.length - 1];
    let deltas: number[] = [];
    for (let i = 0; i < outputLayer.outputSize; i++) {
      deltas.push(outputs[i] - target[i]);
    }

    // Backprop through layers
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer = this.layers[l];
      
      // Accumulate gradients for this layer
      for (let i = 0; i < layer.outputSize; i++) {
        layer.dBiases[i] += deltas[i];
        for (let j = 0; j < layer.inputSize; j++) {
          layer.dWeights[i][j] += deltas[i] * layer.inputs[j];
        }
      }

      // If we are not at the input layer, compute deltas for previous layer
      if (l > 0) {
        const prevLayer = this.layers[l - 1];
        const nextDeltas = Array(prevLayer.outputSize).fill(0);
        
        for (let j = 0; j < prevLayer.outputSize; j++) {
          let errorSum = 0;
          for (let i = 0; i < layer.outputSize; i++) {
            errorSum += deltas[i] * layer.weights[i][j];
          }
          // multiply by derivative of ReLU
          nextDeltas[j] = errorSum * this.reluDerivative(prevLayer.z[j]);
        }
        
        deltas = nextDeltas;
      }
    }
  }

  // Predict class label
  predict(inputs: number[]): { label: number; name: string; confidences: number[] } {
    const outputs = this.forward(inputs);
    let maxIdx = 0;
    let maxVal = -1;
    for (let i = 0; i < outputs.length; i++) {
      if (outputs[i] > maxVal) {
        maxVal = outputs[i];
        maxIdx = i;
      }
    }
    return {
      label: maxIdx,
      name: this.classes[maxIdx],
      confidences: outputs,
    };
  }

  // Compute average Cross Entropy loss on a dataset
  computeLoss(data: { features: number[]; label: number }[]): number {
    let totalLoss = 0;
    for (const sample of data) {
      const outputs = this.forward(sample.features);
      const prob = Math.max(outputs[sample.label], 1e-15); // avoid log(0)
      totalLoss -= Math.log(prob);
    }
    return totalLoss / data.length;
  }

  // Compute accuracy on a dataset
  computeAccuracy(data: { features: number[]; label: number }[]): number {
    let correct = 0;
    for (const sample of data) {
      const pred = this.predict(sample.features);
      if (pred.label === sample.label) {
        correct++;
      }
    }
    return correct / data.length;
  }

  // Asynchronous training function to prevent locking browser UI thread
  async train(
    trainData: { features: number[]; label: number }[],
    valData: { features: number[]; label: number }[],
    hp: ModelHyperparameters,
    onEpochComplete: (progress: TrainingProgress) => void,
    isCancelled: () => boolean
  ): Promise<void> {
    const { learningRate, epochs, batchSize } = hp;

    for (let epoch = 1; epoch <= epochs; epoch++) {
      if (isCancelled()) return;

      // Shuffle training data every epoch
      const shuffled = [...trainData].sort(() => Math.random() - 0.5);

      // Mini-batch gradient descent
      for (let b = 0; b < shuffled.length; b += batchSize) {
        const batch = shuffled.slice(b, b + batchSize);
        
        // 1. Clear layer gradients
        for (const layer of this.layers) {
          layer.clearGradients();
        }

        // 2. Accumulate gradients over batch
        for (const sample of batch) {
          this.backpropagate(sample.features, sample.label);
        }

        // 3. Normalize gradients and apply weights update
        const batchWeight = 1.0 / batch.length;
        for (const layer of this.layers) {
          // Average the gradients
          for (let i = 0; i < layer.outputSize; i++) {
            layer.dBiases[i] *= batchWeight;
            for (let j = 0; j < layer.inputSize; j++) {
              layer.dWeights[i][j] *= batchWeight;
            }
          }
          
          // Update weights and biases with learning rate
          layer.updateWeights(learningRate);
        }
      }

      // Compute statistics
      const trainLoss = this.computeLoss(trainData);
      const trainAcc = this.computeAccuracy(trainData);
      const valLoss = this.computeLoss(valData);
      const valAcc = this.computeAccuracy(valData);

      onEpochComplete({
        epoch,
        trainLoss,
        trainAcc,
        valLoss,
        valAcc,
      });

      // Pause for 1ms to allow UI updates and interrupt handling
      await new Promise(resolve => setTimeout(resolve, 5));
    }
  }

  // Generates a confusion matrix for evaluation
  generateConfusionMatrix(testData: { features: number[]; label: number }[]): number[][] {
    const n = this.classes.length;
    const matrix = Array.from({ length: n }, () => Array(n).fill(0));

    for (const sample of testData) {
      const pred = this.predict(sample.features);
      matrix[sample.label][pred.label]++;
    }

    return matrix;
  }
}
