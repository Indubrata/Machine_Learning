import { useState } from 'react';
import ModelTrainer from './components/ModelTrainer';
import SensorSimulator from './components/SensorSimulator';
import HeterogeneityExplorer from './components/HeterogeneityExplorer';
import AiAnalyst from './components/AiAnalyst';
import { NeuralNetwork } from './ml/model';
import { Cpu, Activity, ShieldAlert, Sparkles, AlertCircle, CheckCircle, Smartphone } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<'trainer' | 'simulator' | 'explorer' | 'ai'>('trainer');
  
  // Shared States
  const [trainedModel, setTrainedModel] = useState<NeuralNetwork | null>(null);
  const [modelMetrics, setModelMetrics] = useState<any | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);
  const [deviceSetup, setDeviceSetup] = useState<string>("");
  const [hyperparameters, setHyperparameters] = useState<any>({
    learningRate: 0.05,
    epochs: 50,
    batchSize: 16,
    hiddenLayers: [32, 16]
  });

  const [activeStream, setActiveStream] = useState<{
    activity: string;
    device: string;
    summary: any;
  } | null>(null);

  const handleModelTrained = (
    model: NeuralNetwork, 
    metrics: any, 
    matrix: number[][], 
    setup: string
  ) => {
    setTrainedModel(model);
    setModelMetrics(metrics);
    setConfusionMatrix(matrix);
    setDeviceSetup(setup);
    
    // Extract layers layout
    const layersLayout = model.layers.slice(0, -1).map(l => l.outputSize);
    setHyperparameters({
      learningRate: model.layers[0].weights[0].length, // approximation or just general log
      epochs: 50, // default placeholder
      batchSize: 16,
      hiddenLayers: layersLayout
    });
  };

  const handleStreamSummary = (summary: any) => {
    if (activeStream) {
      setActiveStream(prev => ({
        ...prev!,
        summary
      }));
    } else {
      setActiveStream({
        activity: 'walking', // default
        device: 'Samsung S4',
        summary
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F5F5F5] flex flex-col font-sans select-none antialiased">
      {/* 1. Master Header Navigation bar */}
      <header className="border-b border-[#333] bg-[#0A0A0A] sticky top-0 z-30 px-6 py-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="text-[10px] tracking-[0.3em] font-mono text-[#FF3E00] uppercase font-bold">Research Project 344 — HHAR Dataset</div>
            <div className="flex items-baseline gap-3 flex-wrap">
              <h1 className="text-4xl md:text-5xl font-serif font-black tracking-tight text-[#F5F5F5] italic">
                HHAR<span className="text-[#FF3E00]">.</span>
              </h1>
              <p className="text-xs uppercase tracking-[0.2em] text-[#F5F5F5]/60 font-light">
                Heterogeneity Human Activity Recognition Classifier
              </p>
            </div>
          </div>

          {/* Model Status Indicator */}
          <div className="flex items-center gap-2 font-mono">
            {trainedModel ? (
              <div className="flex items-center gap-3 bg-[#151515] border border-[#333] px-4 py-2 text-xs">
                <div className="w-2 h-2 rounded-full bg-[#FF3E00] animate-pulse" />
                <div>
                  <span className="block font-bold uppercase tracking-wider text-[#F5F5F5]">Classifier Live</span>
                  <span className="text-[10px] text-slate-400">Val Acc: {(modelMetrics.valAcc * 100).toFixed(1)}%</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-3 bg-[#151515] border border-dashed border-[#333] px-4 py-2 text-xs text-slate-500">
                <div className="w-2 h-2 rounded-full bg-slate-700" />
                <div>
                  <span className="block font-bold uppercase tracking-wider">Awaiting Model</span>
                  <span className="text-[10px]">Playground inactive</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* 2. Primary Tab Selection bar */}
      <nav className="border-b border-[#333] bg-[#0A0A0A]/50 px-6 py-2">
        <div className="max-w-7xl mx-auto flex flex-wrap gap-2">
          {[
            { id: 'trainer', label: '1. Model Trainer', icon: Cpu },
            { id: 'simulator', label: '2. Sensor Simulator', icon: Activity },
            { id: 'explorer', label: '3. Heterogeneity Study', icon: ShieldAlert },
            { id: 'ai', label: '4. AI Consultant', icon: Sparkles },
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2.5 text-[10px] uppercase tracking-widest font-mono transition-all duration-200 cursor-pointer border ${
                  active
                    ? 'bg-[#151515] text-[#FF3E00] border-[#333]'
                    : 'text-slate-400 hover:text-[#F5F5F5] border-transparent hover:bg-[#151515]/30'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${active ? 'text-[#FF3E00]' : ''}`} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* 3. Master Dashboard Body Container */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto space-y-6">
        {activeTab === 'trainer' && (
          <ModelTrainer onModelTrained={handleModelTrained} />
        )}

        {activeTab === 'simulator' && (
          <SensorSimulator 
            trainedModel={trainedModel} 
            onStreamSummary={handleStreamSummary} 
          />
        )}

        {activeTab === 'explorer' && (
          <HeterogeneityExplorer />
        )}

        {activeTab === 'ai' && (
          <AiAnalyst 
            hyperparameters={hyperparameters}
            finalMetrics={modelMetrics}
            confusionMatrix={confusionMatrix}
            deviceSetup={deviceSetup}
            activeStream={activeStream}
          />
        )}
      </main>

      {/* 4. Footer */}
      <footer className="border-t border-[#333] bg-[#0A0A0A] py-8 px-6">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500 text-[10px] font-mono uppercase tracking-widest">
          <span>UCI HAR Laboratory • v0.9 Beta</span>

        </div>
      </footer>
    </div>
  );
}
