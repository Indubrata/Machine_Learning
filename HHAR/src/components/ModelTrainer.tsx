import { useState, useRef, useEffect } from 'react';
import { NeuralNetwork, TrainingProgress, ModelHyperparameters } from '../ml/model';
import { generateDataset, ACTIVITIES } from '../ml/dataset';
import { Play, RotateCcw, AlertTriangle, Cpu, TrendingUp, BarChart2 } from 'lucide-react';

interface ModelTrainerProps {
  onModelTrained: (model: NeuralNetwork, finalMetrics: any, confusionMatrix: number[][], deviceSetup: string) => void;
}

export default function ModelTrainer({ onModelTrained }: ModelTrainerProps) {
  // Hyperparameters
  const [lr, setLr] = useState<number>(0.05);
  const [epochs, setEpochs] = useState<number>(50);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [hiddenNeurons, setHiddenNeurons] = useState<string>("32, 16");

  // Datasets Setup
  const [deviceFilter, setDeviceFilter] = useState<'all' | 'smartphones' | 'watches'>('all');
  const [trainSize, setTrainSize] = useState<number>(400);

  // Training State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [history, setHistory] = useState<TrainingProgress[]>([]);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);
  const [evaluationAcc, setEvaluationAcc] = useState<number | null>(null);

  // Refs for training cancellation
  const cancelTrainingRef = useRef<boolean>(false);
  const modelRef = useRef<NeuralNetwork | null>(null);

  // Generate dataset on load or filter change
  const getDevicesFilter = () => {
    if (deviceFilter === 'smartphones') return ['Samsung S3', 'Samsung S4', 'LG Nexus 4'] as any[];
    if (deviceFilter === 'watches') return ['Galaxy Gear (Watch)'] as any[];
    return ['Samsung S3', 'Samsung S4', 'LG Nexus 4', 'Galaxy Gear (Watch)'] as any[];
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setHistory([]);
    setConfusionMatrix(null);
    setEvaluationAcc(null);
    cancelTrainingRef.current = false;

    // Parse hidden layers
    const hiddenLayers = hiddenNeurons
      .split(',')
      .map(s => parseInt(s.trim()))
      .filter(n => !isNaN(n) && n > 0);

    // Generate training and validation sets
    const allowedDevices = getDevicesFilter();
    const fullTrainData = generateDataset(trainSize, allowedDevices);
    // 20% validation set size
    const valSize = Math.max(50, Math.floor(trainSize * 0.25));
    const fullValData = generateDataset(valSize, allowedDevices);

    // Initialize Neural Network (12 inputs, custom hidden, 6 outputs)
    const nn = new NeuralNetwork(12, hiddenLayers, 6, ACTIVITIES);
    modelRef.current = nn;

    const hp: ModelHyperparameters = {
      learningRate: lr,
      epochs: epochs,
      batchSize: batchSize,
      hiddenLayers: hiddenLayers,
    };

    try {
      await nn.train(
        fullTrainData,
        fullValData,
        hp,
        (progress) => {
          setCurrentEpoch(progress.epoch);
          setHistory(prev => [...prev, progress]);
        },
        () => cancelTrainingRef.current
      );

      if (!cancelTrainingRef.current) {
        // Complete training metrics
        const finalAccuracy = nn.computeAccuracy(fullValData);
        const matrix = nn.generateConfusionMatrix(fullValData);
        setConfusionMatrix(matrix);
        setEvaluationAcc(finalAccuracy);

        const deviceSetupStr = deviceFilter === 'all' 
          ? "All HHAR Devices (Heterogeneous Mix)" 
          : deviceFilter === 'smartphones' 
            ? "Pocket-worn Smartphones Only (S3, S4, Nexus 4)" 
            : "Wrist-worn Smartwatches Only (Galaxy Gear)";

        onModelTrained(nn, {
          trainLoss: nn.computeLoss(fullTrainData),
          valLoss: nn.computeLoss(fullValData),
          trainAcc: nn.computeAccuracy(fullTrainData),
          valAcc: finalAccuracy,
        }, matrix, deviceSetupStr);
      }
    } catch (err) {
      console.error("Training error", err);
    } finally {
      setIsTraining(false);
    }
  };

  const handleCancel = () => {
    cancelTrainingRef.current = true;
    setIsTraining(false);
  };

  // SVG Chart Helper Coordinates
  const getChartPoints = (key: 'trainLoss' | 'valLoss' | 'trainAcc' | 'valAcc', width: number, height: number) => {
    if (history.length === 0) return "";
    
    let maxVal = 1;
    if (key === 'trainLoss' || key === 'valLoss') {
      maxVal = Math.max(...history.map(h => Math.max(h.trainLoss, h.valLoss)), 1.5);
    }

    return history.map((h, i) => {
      const x = (i / (epochs - 1)) * width;
      const val = h[key];
      // invert y coordinate for screen space
      const y = height - (val / maxVal) * height;
      return `${x},${y}`;
    }).join(" ");
  };

  return (
    <div id="model-trainer-section" className="grid grid-cols-1 lg:grid-cols-3 gap-6 font-sans">
      {/* 1. Configuration Panel */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-[#FF3E00]" />
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Network Architecture</h3>
          </div>

          <p className="text-xs text-[#F5F5F5]/75 mb-6 leading-relaxed font-light">
            Configure a Feedforward Neural Network (Multilayer Perceptron) to classify HHAR sensor features. Features include tri-axial mean and standard deviations of smartphone Accelerometers and Gyroscopes.
          </p>

          <div className="space-y-5">
            {/* Devices Selector */}
            <div>
              <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Device Population (Heterogeneity)</label>
              <select
                disabled={isTraining}
                value={deviceFilter}
                onChange={(e) => setDeviceFilter(e.target.value as any)}
                className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] focus:outline-none focus:border-[#FF3E00] font-mono transition-colors"
              >
                <option value="all">All HHAR Devices (Heterogeneous)</option>
                <option value="smartphones">Smartphones Only (Pocket-worn)</option>
                <option value="watches">Smartwatches Only (Wrist-worn)</option>
              </select>
            </div>

            {/* Hidden Layers Input */}
            <div>
              <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Hidden Layers Layout</label>
              <input
                disabled={isTraining}
                type="text"
                value={hiddenNeurons}
                onChange={(e) => setHiddenNeurons(e.target.value)}
                placeholder="e.g. 32, 16"
                className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] font-mono focus:outline-none focus:border-[#FF3E00] transition-colors"
              />
              <span className="text-[9px] text-slate-500 font-mono block mt-1">Comma-separated integers designating neurons per layer.</span>
            </div>

            {/* Learning Rate */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Learning Rate</label>
                <input
                  disabled={isTraining}
                  type="number"
                  step="0.01"
                  min="0.001"
                  max="0.5"
                  value={lr}
                  onChange={(e) => setLr(parseFloat(e.target.value))}
                  className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] font-mono focus:outline-none focus:border-[#FF3E00] transition-colors"
                />
              </div>

              <div>
                <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Epochs</label>
                <input
                  disabled={isTraining}
                  type="number"
                  min="10"
                  max="500"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] font-mono focus:outline-none focus:border-[#FF3E00] transition-colors"
                />
              </div>
            </div>

            {/* Batch Size & Train Dataset Size */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Batch Size</label>
                <select
                  disabled={isTraining}
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] font-mono focus:outline-none focus:border-[#FF3E00] transition-colors"
                >
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                </select>
              </div>

              <div>
                <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5">Samples</label>
                <input
                  disabled={isTraining}
                  type="number"
                  step="50"
                  min="100"
                  max="1000"
                  value={trainSize}
                  onChange={(e) => setTrainSize(parseInt(e.target.value))}
                  className="w-full bg-[#0A0A0A] border border-[#333] rounded-none px-3 py-2.5 text-xs text-[#F5F5F5] font-mono focus:outline-none focus:border-[#FF3E00] transition-colors"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 space-y-2">
          {isTraining ? (
            <button
              onClick={handleCancel}
              className="w-full bg-transparent hover:bg-rose-950/20 border border-rose-500 text-rose-500 font-mono text-xs uppercase tracking-widest py-3.5 px-4 flex items-center justify-center gap-2 transition duration-200"
            >
              <RotateCcw className="w-3.5 h-3.5 animate-spin" />
              Cancel Training
            </button>
          ) : (
            <button
              onClick={handleTrain}
              className="w-full bg-[#FF3E00] hover:bg-[#ff551f] text-white font-mono text-xs uppercase tracking-widest py-3.5 px-4 flex items-center justify-center gap-2 transition duration-200 cursor-pointer"
            >
              <Play className="w-3.5 h-3.5 fill-white" />
              Train Neural Network
            </button>
          )}
        </div>
      </div>

      {/* 2. Real-time Training Curves Chart */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl lg:col-span-2 flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-[#FF3E00]" />
              <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Training Dynamics</h3>
            </div>
            {isTraining && (
              <span className="text-[10px] bg-[#FF3E00]/10 border border-[#FF3E00]/40 text-[#FF3E00] px-3 py-1 font-mono tracking-widest uppercase animate-pulse">
                Epoch {currentEpoch} / {epochs}
              </span>
            )}
          </div>

          {/* Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-[#0A0A0A] p-4 border border-[#333] rounded-none">
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block mb-1">Train Loss</span>
              <span className="text-xl font-mono text-[#F5F5F5] font-semibold">
                {history.length > 0 ? history[history.length - 1].trainLoss.toFixed(4) : "0.0000"}
              </span>
            </div>
            <div className="bg-[#0A0A0A] p-4 border border-[#333] rounded-none">
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block mb-1">Val Loss</span>
              <span className="text-xl font-mono text-[#F5F5F5] font-semibold">
                {history.length > 0 ? history[history.length - 1].valLoss.toFixed(4) : "0.0000"}
              </span>
            </div>
            <div className="bg-[#0A0A0A] p-4 border border-[#333] rounded-none">
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block mb-1">Train Acc</span>
              <span className="text-xl font-mono text-[#FF3E00] font-semibold">
                {history.length > 0 ? `${(history[history.length - 1].trainAcc * 100).toFixed(1)}%` : "0.0%"}
              </span>
            </div>
            <div className="bg-[#0A0A0A] p-4 border border-[#333] rounded-none">
              <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block mb-1">Validation Acc</span>
              <span className="text-xl font-mono text-[#FF3E00] font-bold">
                {history.length > 0 ? `${(history[history.length - 1].valAcc * 100).toFixed(1)}%` : "0.0%"}
              </span>
            </div>
          </div>

          {/* SVG Visualizations of curves */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-52">
            {/* Loss Curve */}
            <div className="bg-[#0A0A0A] border border-[#333] p-4 flex flex-col justify-between h-full relative rounded-none">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-2">Loss Curve</span>
              {history.length > 0 ? (
                <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  <line x1="0" y1="0" x2="300" y2="0" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />
                  <line x1="0" y1="60" x2="300" y2="60" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />
                  <line x1="0" y1="120" x2="300" y2="120" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />
                  
                  {/* Train Loss line */}
                  <polyline
                    fill="none"
                    stroke="#FF3E00"
                    strokeWidth="2"
                    points={getChartPoints('trainLoss', 300, 120)}
                  />
                  {/* Val Loss line */}
                  <polyline
                    fill="none"
                    stroke="#F5F5F5"
                    strokeWidth="1.5"
                    strokeDasharray="4,2"
                    points={getChartPoints('valLoss', 300, 120)}
                  />
                </svg>
              ) : (
                <div className="flex-1 flex items-center justify-center text-[10px] text-slate-600 font-mono uppercase tracking-wider">
                  Awaiting model training...
                </div>
              )}
              <div className="flex justify-between items-center text-[9px] text-slate-500 font-mono mt-2 uppercase tracking-widest">
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#FF3E00] inline-block"></span> Train</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 border-t border-dashed border-[#F5F5F5] inline-block"></span> Validation</span>
              </div>
            </div>

            {/* Accuracy Curve */}
            <div className="bg-[#0A0A0A] border border-[#333] p-4 flex flex-col justify-between h-full relative rounded-none">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-2">Accuracy Curve</span>
              {history.length > 0 ? (
                <svg className="w-full h-full overflow-visible" viewBox="0 0 300 120" preserveAspectRatio="none">
                  {/* Grid Lines representing 0%, 50%, 100% */}
                  <line x1="0" y1="0" x2="300" y2="0" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />
                  <line x1="0" y1="60" x2="300" y2="60" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />
                  <line x1="0" y1="120" x2="300" y2="120" stroke="#262626" strokeWidth="0.5" strokeDasharray="3,3" />

                  {/* Train Acc line */}
                  <polyline
                    fill="none"
                    stroke="#FF3E00"
                    strokeWidth="2"
                    points={getChartPoints('trainAcc', 300, 120)}
                  />
                  {/* Val Acc line */}
                  <polyline
                    fill="none"
                    stroke="#F5F5F5"
                    strokeWidth="1.5"
                    strokeDasharray="4,2"
                    points={getChartPoints('valAcc', 300, 120)}
                  />
                </svg>
              ) : (
                <div className="flex-1 flex items-center justify-center text-[10px] text-slate-600 font-mono uppercase tracking-wider">
                  Awaiting model training...
                </div>
              )}
              <div className="flex justify-between items-center text-[9px] text-slate-500 font-mono mt-2 uppercase tracking-widest">
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-[#FF3E00] inline-block"></span> Train</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 border-t border-dashed border-[#F5F5F5] inline-block"></span> Validation</span>
              </div>
            </div>
          </div>
        </div>

        {/* 3. Confusion Matrix Section */}
        {confusionMatrix && (
          <div className="mt-8 pt-6 border-t border-[#333]">
            <div className="flex items-center gap-2 mb-4">
              <BarChart2 className="w-4 h-4 text-[#FF3E00]" />
              <h4 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Confusion Matrix (Validation)</h4>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full text-center text-xs font-mono text-slate-300">
                <thead>
                  <tr className="border-b border-[#333] text-[9px] text-slate-500 uppercase tracking-widest">
                    <th className="py-2.5 text-left font-light">True \ Pred</th>
                    {ACTIVITIES.map(act => (
                      <th key={act} className="py-2.5 px-1 font-light capitalize">{act}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.map((row, i) => {
                    const rowSum = row.reduce((a, b) => a + b, 0);
                    return (
                      <tr key={ACTIVITIES[i]} className="border-b border-[#262626] hover:bg-[#0A0A0A]/50">
                        <td className="py-3 text-left font-medium capitalize text-slate-400 pr-2">{ACTIVITIES[i]}</td>
                        {row.map((cell, j) => {
                          const percentage = rowSum > 0 ? cell / rowSum : 0;
                          let heatStyle = "bg-[#0A0A0A] text-slate-600";
                          if (percentage > 0.8) heatStyle = "bg-[#FF3E00]/20 text-[#FF3E00] font-bold border border-[#FF3E00]/30";
                          else if (percentage > 0.5) heatStyle = "bg-[#FF3E00]/10 text-[#FF3E00]/80 border border-[#FF3E00]/10";
                          else if (percentage > 0.2) heatStyle = "bg-[#151515] text-[#F5F5F5]";
                          else if (percentage > 0) heatStyle = "bg-[#0A0A0A] text-slate-500";
                          
                          return (
                            <td key={j} className={`py-2 px-1 ${heatStyle}`}>
                              <span className="block font-bold">{cell}</span>
                              <span className="text-[9px] opacity-70">{(percentage * 100).toFixed(0)}%</span>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
