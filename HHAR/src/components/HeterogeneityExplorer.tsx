import { useState } from 'react';
import { NeuralNetwork } from '../ml/model';
import { generateDataset, ACTIVITIES, DEVICES } from '../ml/dataset';
import { ShieldAlert, Compass, Smartphone, RefreshCw, BarChart2, Check, ArrowRight, Play } from 'lucide-react';

export default function HeterogeneityExplorer() {
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [results, setResults] = useState<{
    phoneModel: { onPhone: number; onWatch: number };
    watchModel: { onPhone: number; onWatch: number };
    mixedModel: { onPhone: number; onWatch: number };
  } | null>(null);

  const runEvaluation = async () => {
    setIsEvaluating(true);
    
    // 1. Generate specific populations
    const phoneTrain = generateDataset(300, ['Samsung S3', 'Samsung S4', 'LG Nexus 4']);
    const watchTrain = generateDataset(150, ['Galaxy Gear (Watch)']);
    const mixedTrain = generateDataset(300, ['Samsung S3', 'Samsung S4', 'LG Nexus 4', 'Galaxy Gear (Watch)']);

    // 2. Generate evaluation test sets
    const phoneTest = generateDataset(100, ['Samsung S3', 'Samsung S4', 'LG Nexus 4']);
    const watchTest = generateDataset(100, ['Galaxy Gear (Watch)']);

    // Standard architecture: 1 hidden layer [16]
    const hp = { learningRate: 0.1, epochs: 40, batchSize: 16, hiddenLayers: [16] };

    // Initialize models
    const phoneNN = new NeuralNetwork(12, [16], 6, ACTIVITIES);
    const watchNN = new NeuralNetwork(12, [16], 6, ACTIVITIES);
    const mixedNN = new NeuralNetwork(12, [16], 6, ACTIVITIES);

    // Train them sequentially (fast with lightweight architecture)
    const emptyCallback = () => {};
    const noCancel = () => false;

    try {
      await phoneNN.train(phoneTrain, phoneTrain, hp, emptyCallback, noCancel);
      await watchNN.train(watchTrain, watchTrain, hp, emptyCallback, noCancel);
      await mixedNN.train(mixedTrain, mixedTrain, hp, emptyCallback, noCancel);

      // Evaluate
      setResults({
        phoneModel: {
          onPhone: phoneNN.computeAccuracy(phoneTest),
          onWatch: phoneNN.computeAccuracy(watchTest),
        },
        watchModel: {
          onPhone: watchNN.computeAccuracy(phoneTest),
          onWatch: watchNN.computeAccuracy(watchTest),
        },
        mixedModel: {
          onPhone: mixedNN.computeAccuracy(phoneTest),
          onWatch: mixedNN.computeAccuracy(watchTest),
        }
      });
    } catch (err) {
      console.error("Evaluation error", err);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div id="heterogeneity-explorer-section" className="space-y-6 font-sans">
      {/* Educational Header Banner */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row items-start gap-4">
          <div className="bg-[#FF3E00]/10 p-2 border border-[#FF3E00] text-[#FF3E00] shrink-0">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div className="space-y-1.5 flex-1">
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">The "Heterogeneity Gap" Deep-Dive</h3>
            <p className="text-xs text-slate-400 leading-relaxed max-w-4xl font-light">
              Human Activity Recognition algorithms frequently experience drastic accuracy degradation when tested on a device type different from the one they were trained on. This is due to <strong className="text-slate-200">domain shift</strong>: a smartphone in a pocket records purely linear hips sway, while a smartwatch on a wrist records massive rotations and arm sweeps during the exact same activity (e.g. Walking).
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Visual Comparison: Phone vs Watch sensor profiles */}
        <div className="bg-[#151515] border border-[#333] p-6 shadow-xl space-y-5">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-[#FF3E00]" />
            <h4 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Physical Feature Profiles</h4>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed font-light">
            Observe the difference in sensor parameters for a smartphone in a pocket versus a smartwatch on a swinging wrist. Note how the rotational (Gyroscope) amplitude is almost triple on the wrist!
          </p>

          <div className="space-y-5 bg-[#0A0A0A] p-5 border border-[#333]">
            {/* Linear Accelerometer envelope comparison */}
            <div className="space-y-3">
              <span className="text-[10px] font-bold uppercase tracking-widest text-[#FF3E00] font-mono flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5" /> Accel Std. Dev (m/s²)
              </span>
              
              <div className="grid grid-cols-3 gap-3 items-center">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Pocket Phone</span>
                <div className="col-span-2 h-4 w-full bg-[#151515] border border-[#222] relative">
                  <div className="bg-[#F5F5F5] h-full w-[45%]" style={{ transition: 'width 1s' }}></div>
                  <span className="absolute right-2 top-0.5 text-[9px] font-mono text-[#F5F5F5] font-semibold">2.5 m/s² (Mod)</span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 items-center">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Wrist Watch</span>
                <div className="col-span-2 h-4 w-full bg-[#151515] border border-[#222] relative">
                  <div className="bg-[#FF3E00] h-full w-[65%]" style={{ transition: 'width 1s' }}></div>
                  <span className="absolute right-2 top-0.5 text-[9px] font-mono text-white font-semibold">3.8 m/s² (High)</span>
                </div>
              </div>
            </div>

            {/* Rotational Gyroscope envelope comparison */}
            <div className="space-y-3 pt-3 border-t border-[#333]">
              <span className="text-[10px] font-bold uppercase tracking-widest text-[#FF3E00] font-mono flex items-center gap-1.5">
                <RefreshCw className="w-3.5 h-3.5" /> Gyro Std. Dev (rad/s)
              </span>

              <div className="grid grid-cols-3 gap-3 items-center">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Pocket Phone</span>
                <div className="col-span-2 h-4 w-full bg-[#151515] border border-[#222] relative">
                  <div className="bg-[#F5F5F5] h-full w-[25%]" style={{ transition: 'width 1s' }}></div>
                  <span className="absolute right-2 top-0.5 text-[9px] font-mono text-[#F5F5F5] font-semibold">1.2 rad/s (Low)</span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3 items-center">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Wrist Watch</span>
                <div className="col-span-2 h-4 w-full bg-[#151515] border border-[#222] relative">
                  <div className="bg-[#FF3E00] h-full w-[85%]" style={{ transition: 'width 1s' }}></div>
                  <span className="absolute right-2 top-0.5 text-[9px] font-mono text-white font-bold">3.4 rad/s (High)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="text-[10px] text-slate-500 leading-relaxed font-mono uppercase tracking-wider">
            Rotational features are scaled up by over 280% on the wrist. If not calibrated with multi-placement domain data, a pocket-trained classifier will immediately saturate its decision gates.
          </div>
        </div>

        {/* Transfer Learning Evaluation Simulator */}
        <div className="bg-[#151515] border border-[#333] p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <RefreshCw className="w-4 h-4 text-[#FF3E00]" />
              <h4 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Generalization Test</h4>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed mb-5 font-light">
              Train three distinct neural networks simultaneously under different device populations, and evaluate their cross-compatibility performance on Phone vs. Watch test datasets.
            </p>

            {results ? (
              <div className="space-y-4">
                <table className="w-full text-xs text-left font-mono text-slate-300">
                  <thead>
                    <tr className="border-b border-[#333] text-[9px] text-slate-500 uppercase tracking-wider">
                      <th className="py-2.5 font-light">Model (Training Set)</th>
                      <th className="py-2.5 text-center text-[#F5F5F5] font-light">Tested on Phone</th>
                      <th className="py-2.5 text-center text-[#FF3E00] font-light">Tested on Watch</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-[#222] hover:bg-[#0A0A0A]/30">
                      <td className="py-3.5 flex items-center gap-1.5 font-medium text-slate-400">
                        <Smartphone className="w-3.5 h-3.5 text-slate-400" /> Phone Only (Pocket)
                      </td>
                      <td className="py-3.5 text-center text-[#F5F5F5] font-bold">
                        {(results.phoneModel.onPhone * 100).toFixed(0)}%
                      </td>
                      <td className="py-3.5 text-center text-[#FF3E00] font-bold">
                        {(results.phoneModel.onWatch * 100).toFixed(0)}% <span className="text-[8px] font-medium block text-slate-500 uppercase tracking-widest mt-0.5">Shift!</span>
                      </td>
                    </tr>
                    <tr className="border-b border-[#222] hover:bg-[#0A0A0A]/30">
                      <td className="py-3.5 flex items-center gap-1.5 font-medium text-slate-400">
                        <RefreshCw className="w-3.5 h-3.5 text-slate-400" /> Watch Only (Wrist)
                      </td>
                      <td className="py-3.5 text-center text-[#FF3E00] font-bold">
                        {(results.watchModel.onPhone * 100).toFixed(0)}% <span className="text-[8px] font-medium block text-slate-500 uppercase tracking-widest mt-0.5">Shift!</span>
                      </td>
                      <td className="py-3.5 text-center text-[#F5F5F5] font-bold">
                        {(results.watchModel.onWatch * 100).toFixed(0)}%
                      </td>
                    </tr>
                    <tr className="hover:bg-[#0A0A0A]/30">
                      <td className="py-3.5 flex items-center gap-1.5 font-bold text-slate-200">
                        <Check className="w-3.5 h-3.5 text-[#FF3E00]" /> Mixed Population (All)
                      </td>
                      <td className="py-3.5 text-center text-[#FF3E00] font-bold text-sm">
                        {(results.mixedModel.onPhone * 100).toFixed(0)}%
                      </td>
                      <td className="py-3.5 text-center text-[#FF3E00] font-bold text-sm">
                        {(results.mixedModel.onWatch * 100).toFixed(0)}%
                      </td>
                    </tr>
                  </tbody>
                </table>

                <div className="bg-[#FF3E00]/5 border border-[#FF3E00]/20 text-[#FF3E00] text-xs p-4 flex items-start gap-3 leading-relaxed">
                  <ArrowRight className="w-4 h-4 shrink-0 mt-0.5" />
                  <span className="font-light text-slate-300 text-[11px]">
                    <strong className="text-[#FF3E00] font-mono uppercase tracking-wider block mb-1">Empirical Result</strong>
                    The <strong className="text-[#FF3E00]">Mixed Model</strong> generalizes successfully across both physical environments because its training dataset contained heterogeneous variance from both devices. Placement diversity is a core design criterion for robust sensor intelligence!
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-14 text-slate-600 border border-dashed border-[#333]">
                <RefreshCw className="w-8 h-8 text-slate-800 mx-auto mb-2 animate-pulse" />
                <p className="text-[10px] font-mono uppercase tracking-widest">Cross-device training is inactive</p>
              </div>
            )}
          </div>

          <button
            onClick={runEvaluation}
            disabled={isEvaluating}
            className="w-full mt-6 bg-[#FF3E00] hover:bg-[#ff551f] disabled:bg-[#333] disabled:text-slate-500 text-white text-[11px] font-mono uppercase tracking-widest py-3 flex items-center justify-center gap-2 transition duration-200 cursor-pointer"
          >
            {isEvaluating ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Optimizing 3 Populations...
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-white" />
                Run Cross-Device Generalization Test
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
