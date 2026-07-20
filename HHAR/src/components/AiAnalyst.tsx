import { useState } from 'react';
import { Cpu, Activity, Sparkles, Send, RefreshCw, GraduationCap } from 'lucide-react';

interface AiAnalystProps {
  hyperparameters: any;
  finalMetrics: any;
  confusionMatrix: number[][] | null;
  deviceSetup: string;
  activeStream: {
    activity: string;
    device: string;
    summary: any;
  } | null;
}

export default function AiAnalyst({
  hyperparameters,
  finalMetrics,
  confusionMatrix,
  deviceSetup,
  activeStream
}: AiAnalystProps) {
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisText, setAnalysisText] = useState<string>("");
  const [activeTab, setActiveTab] = useState<'model' | 'coach'>('model');

  const requestModelAnalysis = async () => {
    setLoading(true);
    setAnalysisText("");
    try {
      const response = await fetch("/api/analyze-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hyperparameters,
          finalMetrics,
          confusionMatrix,
          deviceSetup
        })
      });

      const data = await response.json();
      if (response.ok) {
        setAnalysisText(data.analysis);
      } else {
        setAnalysisText(`Error: ${data.error || "Failed to analyze model configuration."}`);
      }
    } catch (err: any) {
      setAnalysisText(`Failed to connect to the analysis service: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const requestStreamAdvice = async () => {
    if (!activeStream) return;
    setLoading(true);
    setAnalysisText("");
    try {
      const response = await fetch("/api/analyze-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          activity: activeStream.activity,
          device: activeStream.device,
          samplesSummary: activeStream.summary
        })
      });

      const data = await response.json();
      if (response.ok) {
        setAnalysisText(data.advice);
      } else {
        setAnalysisText(`Error: ${data.error || "Failed to retrieve coaching advice."}`);
      }
    } catch (err: any) {
      setAnalysisText(`Failed to connect to the coaching service: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Safe and clean simple Markdown renderer for the AI outputs
  const parseMarkdownToJsx = (text: string) => {
    if (!text) return null;

    const lines = text.split("\n");
    return lines.map((line, i) => {
      // Check for headings
      if (line.startsWith("### ")) {
        return <h5 key={i} className="text-xs uppercase tracking-[0.15em] font-mono font-bold text-[#FF3E00] mt-6 mb-2">{line.replace("### ", "")}</h5>;
      }
      if (line.startsWith("## ")) {
        return <h4 key={i} className="text-sm uppercase tracking-[0.2em] font-bold text-[#F5F5F5] mt-8 mb-3 border-b border-[#333] pb-1.5 font-sans">{line.replace("## ", "")}</h4>;
      }
      if (line.startsWith("# ")) {
        return <h3 key={i} className="text-lg font-serif font-black italic text-[#FF3E00] mt-10 mb-4 border-b-2 border-[#FF3E00]/20 pb-2">{line.replace("# ", "")}</h3>;
      }
      // Check for bullet points
      if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
        const cleanLine = line.trim().substring(2);
        return (
          <li key={i} className="ml-5 list-disc text-xs text-slate-300 leading-relaxed my-1 font-light">
            {formatBoldText(cleanLine)}
          </li>
        );
      }
      // Bullet lists numbered
      if (/^\d+\.\s/.test(line.trim())) {
        const cleanLine = line.trim().replace(/^\d+\.\s/, "");
        return (
          <li key={i} className="ml-5 list-decimal text-xs text-slate-300 leading-relaxed my-1 font-light">
            {formatBoldText(cleanLine)}
          </li>
        );
      }
      // Standard lines
      if (line.trim() === "") return <div key={i} className="h-3"></div>;

      return <p key={i} className="text-xs text-slate-300 leading-relaxed my-2.5 font-light">{formatBoldText(line)}</p>;
    });
  };

  // Help format inline bold markdown elements e.g. **text**
  const formatBoldText = (str: string) => {
    const parts = str.split(/\*\*([^*]+)\*\*/g);
    if (parts.length === 1) return str;
    return parts.map((part, index) => {
      return index % 2 === 1 ? <strong key={index} className="text-[#FF3E00] font-bold">{part}</strong> : part;
    });
  };

  return (
    <div id="ai-analyst-section" className="grid grid-cols-1 lg:grid-cols-3 gap-6 font-sans">
      {/* Selector Side Menu */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-[#FF3E00]" />
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">AI Insights Portal</h3>
          </div>

          <p className="text-xs text-slate-400 mb-6 leading-relaxed font-light">
            Bridge your sensory experiments with Llama AI to analyze your ML results or receive specialized posture training tips.
          </p>

          {/* Tab togglers */}
          <div className="space-y-3">
            <button
              onClick={() => { setActiveTab('model'); setAnalysisText(""); }}
              className={`w-full p-4 text-left rounded-none border text-xs flex items-center gap-3 transition duration-200 cursor-pointer ${
                activeTab === 'model'
                  ? 'bg-[#FF3E00]/10 border-[#FF3E00] text-[#FF3E00]'
                  : 'bg-[#0A0A0A] border-[#333] text-slate-400 hover:text-[#F5F5F5]'
              }`}
            >
              <Cpu className="w-4 h-4 shrink-0 text-[#FF3E00]" />
              <div>
                <strong className="block font-mono uppercase tracking-wider text-[11px]">Scientific ML Analyst</strong>
                <span className="text-[10px] opacity-75 block mt-0.5 font-light">Evaluate loss, accuracy, and confusion layers.</span>
              </div>
            </button>

            <button
              onClick={() => { setActiveTab('coach'); setAnalysisText(""); }}
              className={`w-full p-4 text-left rounded-none border text-xs flex items-center gap-3 transition duration-200 cursor-pointer ${
                activeTab === 'coach'
                  ? 'bg-[#FF3E00]/10 border-[#FF3E00] text-[#FF3E00]'
                  : 'bg-[#0A0A0A] border-[#333] text-slate-400 hover:text-[#F5F5F5]'
              }`}
            >
              <Activity className="w-4 h-4 shrink-0 text-[#FF3E00]" />
              <div>
                <strong className="block font-mono uppercase tracking-wider text-[11px]">Wearable Motion Coach</strong>
                <span className="text-[10px] opacity-75 block mt-0.5 font-light">Real-time biomechanics feedback on active streams.</span>
              </div>
            </button>
          </div>
        </div>

        <div className="mt-8 pt-5 border-t border-[#333]">
          {activeTab === 'model' ? (
            <div>
              {!finalMetrics ? (
                <div className="text-[10px] text-[#FF3E00] font-mono flex items-center gap-1.5 mb-3 bg-[#FF3E00]/5 border border-[#FF3E00]/20 p-3 uppercase tracking-widest text-center justify-center">
                  <span>Awaiting Model Training</span>
                </div>
              ) : (
                <button
                  onClick={requestModelAnalysis}
                  disabled={loading}
                  className="w-full bg-[#FF3E00] hover:bg-[#ff551f] disabled:bg-[#333] disabled:text-slate-500 text-white text-[11px] font-mono uppercase tracking-widest py-3 flex items-center justify-center gap-2 transition duration-200 cursor-pointer"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Consulting Llama...
                    </>
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5" />
                      Analyze ML Performance
                    </>
                  )}
                </button>
              )}
            </div>
          ) : (
            <div>
              {!activeStream ? (
                <div className="text-[10px] text-[#FF3E00] font-mono flex items-center gap-1.5 mb-3 bg-[#FF3E00]/5 border border-[#FF3E00]/20 p-3 uppercase tracking-widest text-center justify-center">
                  <span>Start active stream first</span>
                </div>
              ) : (
                <button
                  onClick={requestStreamAdvice}
                  disabled={loading}
                  className="w-full bg-[#FF3E00] hover:bg-[#ff551f] disabled:bg-[#333] disabled:text-slate-500 text-white text-[11px] font-mono uppercase tracking-widest py-3 flex items-center justify-center gap-2 transition duration-200 cursor-pointer"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Consulting Expert...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 fill-white" />
                      Request Movement Coaching
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Analytical Document Viewer */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl lg:col-span-2 flex flex-col justify-between min-h-[480px]">
        <div className="flex-1 flex flex-col">
          <div className="flex items-center gap-2 mb-4 border-b border-[#333] pb-3">
            <GraduationCap className="w-4 h-4 text-slate-400" />
            <h4 className="text-xs uppercase tracking-[0.2em] font-bold text-slate-300">
              {activeTab === 'model' ? 'Academic ML Evaluation Report' : 'Biomechanics Athletic Log'}
            </h4>
          </div>

          <div className="flex-1 bg-[#0A0A0A] border border-[#222] p-6 overflow-y-auto max-h-[500px]">
            {analysisText ? (
              <div className="space-y-1 select-text">
                {parseMarkdownToJsx(analysisText)}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-600 text-center py-16">
                <Sparkles className="w-8 h-8 mb-3 text-slate-800" />
                <h5 className="text-[10px] font-mono uppercase tracking-widest mb-1 text-slate-500">Document Reader Idle</h5>
                <p className="text-[11px] text-slate-600 max-w-sm leading-relaxed font-light">
                  {activeTab === 'model'
                    ? 'Click "Analyze ML Performance" to send your final validation accuracy, confusion matrix, and training dynamics to Llama for expert review.'
                    : 'Click "Request Movement Coaching" while streaming live sensor signals to receive customized movement guidance and diagnostic feedback.'}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="mt-5 text-[9px] text-slate-500 font-mono flex items-center gap-2 bg-[#0A0A0A] p-3 border border-[#222] uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5 text-[#FF3E00]" />
          <span>Report pipeline: <strong>llama-3.3-70b-versatile</strong> via secure server-side interactions proxy</span>
        </div>
      </div>
    </div>
  );
}
