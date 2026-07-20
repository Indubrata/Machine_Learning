import { useState, useEffect, useRef } from 'react';
import { NeuralNetwork } from '../ml/model';
import { generateRawTimeSeries, extractFeaturesFromRawSamples, RawSensorSample, DEVICES, ACTIVITIES, ActivityType, DeviceType } from '../ml/dataset';
import { Play, Square, Smartphone, Compass, Activity, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

interface SensorSimulatorProps {
  trainedModel: NeuralNetwork | null;
  onStreamSummary: (summary: any) => void;
}

export default function SensorSimulator({ trainedModel, onStreamSummary }: SensorSimulatorProps) {
  const [selectedActivity, setSelectedActivity] = useState<ActivityType>('walking');
  const [selectedDevice, setSelectedDevice] = useState<DeviceType>('Samsung S4');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [sensorData, setSensorData] = useState<RawSensorSample[]>([]);
  const [prediction, setPrediction] = useState<{ label: number; name: string; confidences: number[] } | null>(null);

  // Mobile Web Sensor Connection State
  const [useMobileSensors, setUseMobileSensors] = useState<boolean>(false);
  const [mobileError, setMobileError] = useState<string | null>(null);

  // Refs for animation/interval loops
  const streamIntervalRef = useRef<any>(null);
  const dataWindowRef = useRef<RawSensorSample[]>([]);
  const timeRef = useRef<number>(0);

  // Stop stream on unmount
  useEffect(() => {
    return () => stopStream();
  }, []);

  const startStream = () => {
    setIsStreaming(true);
    setMobileError(null);
    setSensorData([]);
    dataWindowRef.current = [];
    timeRef.current = 0;

    if (useMobileSensors) {
      startMobileSensors();
    } else {
      startSimulatedStream();
    }
  };

  const stopStream = () => {
    setIsStreaming(false);
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }
    window.removeEventListener('devicemotion', handleDeviceMotion);
  };

  const startSimulatedStream = () => {
    // We generate fresh timeseries segments of HHAR wave patterns
    const rawFeed = generateRawTimeSeries(selectedActivity, selectedDevice, 30); // 30s stream buffer
    let index = 0;

    streamIntervalRef.current = setInterval(() => {
      if (index >= rawFeed.length) {
        index = 0; // loop
      }

      const sample = rawFeed[index++];
      const rolling = [...dataWindowRef.current, sample];
      
      // Keep trailing 100 samples for the rolling graph window (~2 seconds at 50Hz)
      if (rolling.length > 100) {
        rolling.shift();
      }

      dataWindowRef.current = rolling;
      setSensorData(rolling);

      // Run inference every 15 samples (approx 300ms)
      if (index % 15 === 0 && rolling.length >= 20) {
        runInference(rolling);
      }
    }, 20); // 50 Hz tickrate
  };

  const startMobileSensors = () => {
    // Check if device motion exists
    if (!window.DeviceMotionEvent) {
      setMobileError("DeviceMotion is not supported on this browser.");
      setUseMobileSensors(false);
      setIsStreaming(false);
      return;
    }

    // Request iOS permissions if needed
    const requestPermission = (DeviceMotionEvent as any).requestPermission;
    if (typeof requestPermission === 'function') {
      requestPermission()
        .then((permissionState: string) => {
          if (permissionState === 'granted') {
            window.addEventListener('devicemotion', handleDeviceMotion);
            setupMobileSamplingInterval();
          } else {
            setMobileError("Permission to access motion sensors was denied.");
            setUseMobileSensors(false);
            setIsStreaming(false);
          }
        })
        .catch((err: any) => {
          setMobileError("Could not request motion permissions: " + err.message);
          setUseMobileSensors(false);
          setIsStreaming(false);
        });
    } else {
      // Standard browsers
      window.addEventListener('devicemotion', handleDeviceMotion);
      setupMobileSamplingInterval();
    }
  };

  const mobileAccelRef = useRef({ ax: 0, ay: 0, az: 0 });
  const mobileGyroRef = useRef({ gx: 0, gy: 0, gz: 0 });

  const handleDeviceMotion = (event: DeviceMotionEvent) => {
    // Extract raw accelerometer
    const acc = event.accelerationIncludingGravity || { x: 0, y: 0, z: 0 };
    // Extract raw gyroscope
    const rot = event.rotationRate || { alpha: 0, beta: 0, gamma: 0 };

    mobileAccelRef.current = {
      ax: acc.x || 0,
      ay: acc.y || 0,
      az: acc.z || 0
    };

    // Convert gyro to radians/sec (DeviceMotion offers degrees/sec usually)
    const degToRad = Math.PI / 180;
    mobileGyroRef.current = {
      gx: (rot.alpha || 0) * degToRad,
      gy: (rot.beta || 0) * degToRad,
      gz: (rot.gamma || 0) * degToRad
    };
  };

  const setupMobileSamplingInterval = () => {
    let sampleCounter = 0;
    streamIntervalRef.current = setInterval(() => {
      const sample: RawSensorSample = {
        timestamp: Date.now(),
        ax: mobileAccelRef.current.ax,
        ay: mobileAccelRef.current.ay,
        az: mobileAccelRef.current.az,
        gx: mobileGyroRef.current.gx,
        gy: mobileGyroRef.current.gy,
        gz: mobileGyroRef.current.gz
      };

      const rolling = [...dataWindowRef.current, sample];
      if (rolling.length > 100) {
        rolling.shift();
      }

      dataWindowRef.current = rolling;
      setSensorData(rolling);

      sampleCounter++;
      // Run inference every 15 samples (~300ms)
      if (sampleCounter % 15 === 0 && rolling.length >= 20) {
        runInference(rolling);
      }
    }, 20); // 50 Hz mobile sampling
  };

  const runInference = (samples: RawSensorSample[]) => {
    const features = extractFeaturesFromRawSamples(samples);
    
    // Pass summary back to parent for Gemini context
    onStreamSummary({
      mean_ax: features[0],
      mean_ay: features[1],
      mean_az: features[2],
      std_ax: features[3],
      std_ay: features[4],
      std_az: features[5],
      mean_gx: features[6],
      mean_gy: features[7],
      mean_gz: features[8],
      std_gx: features[9],
      std_gy: features[10],
      std_gz: features[11],
    });

    if (trainedModel) {
      const pred = trainedModel.predict(features);
      setPrediction(pred);
    }
  };

  // Convert array of points to SVG polyline coordinates
  const getRollingPoints = (key: 'ax' | 'ay' | 'az' | 'gx' | 'gy' | 'gz', scale: number, height: number) => {
    if (sensorData.length === 0) return "";
    const len = sensorData.length;
    const padding = 5;
    const midY = height / 2;

    return sensorData.map((s, i) => {
      const x = (i / 99) * 300; // fit into 300 width
      const val = s[key];
      // map to visual canvas height
      const y = midY - val * scale;
      return `${x},${y}`;
    }).join(" ");
  };

  return (
    <div id="sensor-simulator-section" className="grid grid-cols-1 lg:grid-cols-3 gap-6 font-sans">
      {/* Waveform Visualization Panel */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl lg:col-span-2 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#FF3E00]" />
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Active Sensor Stream</h3>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            {/* Real device connection toggle */}
            <button
              onClick={() => {
                if (isStreaming) stopStream();
                setUseMobileSensors(!useMobileSensors);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono border transition duration-200 cursor-pointer uppercase tracking-wider ${
                useMobileSensors 
                  ? 'bg-[#FF3E00]/10 border-[#FF3E00] text-[#FF3E00]' 
                  : 'bg-[#0A0A0A] border-[#333] text-slate-400 hover:text-[#F5F5F5]'
              }`}
            >
              <Smartphone className="w-3.5 h-3.5" />
              {useMobileSensors ? 'Sensor: Mobile' : 'Use Live Phone'}
            </button>

            {isStreaming ? (
              <button
                onClick={stopStream}
                className="bg-transparent border border-[#FF3E00] text-[#FF3E00] hover:bg-[#FF3E00]/10 text-[10px] font-mono uppercase tracking-widest py-1.5 px-3.5 flex items-center gap-1.5 transition duration-200 cursor-pointer"
              >
                <Square className="w-3 h-3 fill-[#FF3E00]" />
                Stop
              </button>
            ) : (
              <button
                onClick={startStream}
                className="bg-[#FF3E00] hover:bg-[#ff551f] text-white text-[10px] font-mono uppercase tracking-widest py-1.5 px-3.5 flex items-center gap-1.5 transition duration-200 cursor-pointer"
              >
                <Play className="w-3 h-3 fill-white" />
                Start Stream
              </button>
            )}
          </div>
        </div>

        {mobileError && (
          <div className="bg-rose-950/20 border border-rose-900/60 text-rose-400 p-3 text-xs font-mono">
            <span>{mobileError}</span>
          </div>
        )}

        {/* Dropdowns for simulated profiles */}
        {!useMobileSensors && (
          <div className="grid grid-cols-2 gap-4 bg-[#0A0A0A] p-4 border border-[#333]">
            <div>
              <label className="block text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-1.5">Target Activity</label>
              <select
                disabled={isStreaming}
                value={selectedActivity}
                onChange={(e) => setSelectedActivity(e.target.value as ActivityType)}
                className="w-full bg-[#151515] border border-[#333] rounded-none px-2.5 py-1.5 text-xs text-[#F5F5F5] capitalize focus:outline-none focus:border-[#FF3E00] font-mono"
              >
                {ACTIVITIES.map(act => (
                  <option key={act} value={act}>{act}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[9px] font-mono text-slate-500 uppercase tracking-widest mb-1.5">Device Profile</label>
              <select
                disabled={isStreaming}
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.target.value as DeviceType)}
                className="w-full bg-[#151515] border border-[#333] rounded-none px-2.5 py-1.5 text-xs text-[#F5F5F5] focus:outline-none focus:border-[#FF3E00] font-mono"
              >
                {DEVICES.map(dev => (
                  <option key={dev} value={dev}>{dev}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* Double Waveform Visualization (Accel & Gyro) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Accelerometer Waveform */}
          <div className="bg-[#0A0A0A] border border-[#333] p-4 flex flex-col justify-between h-52">
            <div className="flex justify-between items-center text-xs text-slate-400 font-mono">
              <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider"><Compass className="w-3.5 h-3.5 text-[#FF3E00]" /> Accel (m/s²)</span>
              <div className="flex gap-2 text-[9px] tracking-wider uppercase font-semibold">
                <span className="text-[#FF3E00]">X</span>
                <span className="text-[#F5F5F5]">Y</span>
                <span className="text-slate-500">Z</span>
              </div>
            </div>
            
            {sensorData.length > 0 ? (
              <svg className="w-full h-32 overflow-visible" viewBox="0 0 300 80" preserveAspectRatio="none">
                <line x1="0" y1="40" x2="300" y2="40" stroke="#1c1c1c" strokeWidth="1" />
                <polyline fill="none" stroke="#FF3E00" strokeWidth="1.5" points={getRollingPoints('ax', 3, 80)} />
                <polyline fill="none" stroke="#F5F5F5" strokeWidth="1.5" points={getRollingPoints('ay', 3, 80)} />
                <polyline fill="none" stroke="#666666" strokeWidth="1.5" points={getRollingPoints('az', 3, 80)} />
              </svg>
            ) : (
              <div className="flex-1 flex items-center justify-center text-[10px] text-slate-600 font-mono uppercase tracking-widest">
                Awaiting connection...
              </div>
            )}

            <div className="text-[9px] text-slate-500 font-mono flex justify-between uppercase tracking-wider">
              <span>Trailing: 2s</span>
              <span>Rate: 50Hz</span>
            </div>
          </div>

          {/* Gyroscope Waveform */}
          <div className="bg-[#0A0A0A] border border-[#333] p-4 flex flex-col justify-between h-52">
            <div className="flex justify-between items-center text-xs text-slate-400 font-mono">
              <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider"><RefreshCw className="w-3.5 h-3.5 text-[#FF3E00]" /> Gyro (rad/s)</span>
              <div className="flex gap-2 text-[9px] tracking-wider uppercase font-semibold">
                <span className="text-[#FF3E00]">X</span>
                <span className="text-[#F5F5F5]">Y</span>
                <span className="text-slate-500">Z</span>
              </div>
            </div>
            
            {sensorData.length > 0 ? (
              <svg className="w-full h-32 overflow-visible" viewBox="0 0 300 80" preserveAspectRatio="none">
                <line x1="0" y1="40" x2="300" y2="40" stroke="#1c1c1c" strokeWidth="1" />
                <polyline fill="none" stroke="#FF3E00" strokeWidth="1.5" points={getRollingPoints('gx', 6, 80)} />
                <polyline fill="none" stroke="#F5F5F5" strokeWidth="1.5" points={getRollingPoints('gy', 6, 80)} />
                <polyline fill="none" stroke="#666666" strokeWidth="1.5" points={getRollingPoints('gz', 6, 80)} />
              </svg>
            ) : (
              <div className="flex-1 flex items-center justify-center text-[10px] text-slate-600 font-mono uppercase tracking-widest">
                Awaiting connection...
              </div>
            )}

            <div className="text-[9px] text-slate-500 font-mono flex justify-between uppercase tracking-wider">
              <span>Trailing: 2s</span>
              <span>Rate: 50Hz</span>
            </div>
          </div>
        </div>
      </div>

      {/* Live Prediction Display */}
      <div className="bg-[#151515] border border-[#333] p-6 shadow-xl flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-4 h-4 text-[#FF3E00]" />
            <h3 className="text-xs uppercase tracking-[0.2em] font-bold text-[#FF3E00]">Inference Predictor</h3>
          </div>

          {!trainedModel && (
            <div className="bg-[#FF3E00]/5 border border-[#FF3E00]/20 p-4 text-xs text-[#FF3E00] space-y-2 mb-4">
              <p className="font-mono uppercase tracking-widest font-bold">No Model Trained Yet</p>
              <p className="text-[11px] text-slate-400 leading-relaxed font-light">
                The simulator is streaming raw values, but the classifier has not been trained. Go to the <strong>Model Trainer</strong> tab above to configure and train your neural network first.
              </p>
            </div>
          )}

          {trainedModel && isStreaming && prediction ? (
            <div className="space-y-6">
              <div className="bg-[#0A0A0A] border border-[#333] p-5 text-center">
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block">Classified Activity</span>
                <span className="text-3xl font-serif font-black italic text-[#FF3E00] capitalize block mt-2">
                  {prediction.name}
                </span>
                <span className="text-[10px] text-[#F5F5F5]/60 font-mono mt-1.5 inline-block uppercase tracking-wider">
                  Confidence: {(prediction.confidences[prediction.label] * 100).toFixed(1)}%
                </span>
                <div className="mt-4 text-[10px] font-mono uppercase tracking-wider border-t border-[#333] pt-3 text-slate-500 flex justify-center gap-4">
                  <span>True: <strong className="capitalize text-slate-300">{useMobileSensors ? "Live" : selectedActivity}</strong></span>
                  <span>Match: {prediction.name === selectedActivity || useMobileSensors ? (
                    <span className="text-[#FF3E00] font-bold">Yes</span>
                  ) : (
                    <span className="text-slate-400">No</span>
                  )}</span>
                </div>
              </div>

              {/* Confidence bars list */}
              <div className="space-y-3">
                <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest block">Model Confidences</span>
                {ACTIVITIES.map((act, i) => {
                  const conf = prediction.confidences[i] || 0;
                  const pct = (conf * 100).toFixed(0);
                  const isPredicted = prediction.label === i;
                  
                  return (
                    <div key={act} className="group flex items-center justify-between text-xs font-mono">
                      <span className={`capitalize w-20 tracking-tight ${isPredicted ? 'text-[#FF3E00] font-bold' : 'text-slate-400'}`}>
                        {act}
                      </span>
                      <div className="flex-1 h-[1px] bg-[#333] mx-3"></div>
                      <div className="w-24 h-4 bg-[#0A0A0A] relative border border-[#222]">
                        <div
                          className={`absolute left-0 top-0 h-full transition-all duration-300 ${isPredicted ? 'bg-[#FF3E00]' : 'bg-[#F5F5F5]/30'}`}
                          style={{ width: `${pct}%` }}
                        ></div>
                      </div>
                      <span className={`w-8 text-right text-[10px] ${isPredicted ? 'text-[#FF3E00] font-bold' : 'text-slate-500'}`}>
                        {pct}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-600 border border-dashed border-[#333]">
              <Activity className="w-8 h-8 text-slate-800 mx-auto mb-2 animate-pulse" />
              <p className="text-[10px] font-mono uppercase tracking-widest">Stream inactive</p>
            </div>
          )}
        </div>

        <div className="mt-8 text-[9px] text-slate-500 font-mono leading-relaxed bg-[#0A0A0A] p-4 border border-[#333] uppercase tracking-wider">
          <strong>Process pipeline:</strong> Buffers signals at 50Hz, extracts 12 statistical descriptors over rolling 2s envelopes, and runs local feedforward predictions with millisecond response time.
        </div>
      </div>
    </div>
  );
}
