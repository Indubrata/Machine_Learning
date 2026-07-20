/**
 * HHAR (Heterogeneity Human Activity Recognition) Dataset Simulator and Feature Generator
 * 
 * Activities: 'biking', 'sitting', 'standing', 'walking', 'stairup', 'stairdown'
 * Smartphone/Smartwatch Heterogeneity modeled:
 * - Samsung Galaxy S3 (GT-I9300) - Pocket-worn, standard noise
 * - Samsung Galaxy S4 (GT-I9500) - Pocket-worn, high precision, low noise
 * - LG Nexus 4 - Pocket-worn, moderate bias, high frequency
 * - Samsung Galaxy Gear (Watch) - Wrist-worn, high gyro rotation, higher noise
 */

export type ActivityType = 'biking' | 'sitting' | 'standing' | 'walking' | 'stairup' | 'stairdown';
export type DeviceType = 'Samsung S3' | 'Samsung S4' | 'LG Nexus 4' | 'Galaxy Gear (Watch)';

export const ACTIVITIES: ActivityType[] = ['biking', 'sitting', 'standing', 'walking', 'stairup', 'stairdown'];
export const DEVICES: DeviceType[] = ['Samsung S3', 'Samsung S4', 'LG Nexus 4', 'Galaxy Gear (Watch)'];

export interface FeatureVector {
  features: number[]; // 12 features: [mean_ax, mean_ay, mean_az, std_ax, std_ay, std_az, mean_gx, mean_gy, mean_gz, std_gx, std_gy, std_gz]
  label: number; // 0 to 5
  activity: ActivityType;
  device: DeviceType;
}

export interface RawSensorSample {
  timestamp: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
}

// Stats profile for each activity (baseline)
interface ActivityProfile {
  accelMean: [number, number, number]; // [x, y, z]
  accelStd: [number, number, number];
  gyroMean: [number, number, number];
  gyroStd: [number, number, number];
  frequency: number; // Dominant frequency of cyclic motion (Hz)
}

const ACTIVITY_PROFILES: Record<ActivityType, ActivityProfile> = {
  sitting: {
    accelMean: [0.1, 9.7, 0.5], // y-axis is mostly gravity (facing up/forward in pocket)
    accelStd: [0.03, 0.03, 0.03],
    gyroMean: [0.0, 0.0, 0.0],
    gyroStd: [0.01, 0.01, 0.01],
    frequency: 0,
  },
  standing: {
    accelMean: [0.2, 9.6, 0.8], // slight postural tilt
    accelStd: [0.08, 0.09, 0.08], // slight micro-movements
    gyroMean: [0.01, -0.01, 0.0],
    gyroStd: [0.03, 0.04, 0.03],
    frequency: 0,
  },
  walking: {
    accelMean: [0.5, 9.4, 1.2],
    accelStd: [1.8, 2.5, 1.5], // large swing in standard deviations
    gyroMean: [0.1, 0.2, -0.1],
    gyroStd: [0.8, 1.2, 0.9],
    frequency: 1.8, // standard walking gait frequency (~1.8 Hz)
  },
  biking: {
    accelMean: [1.2, 8.5, 2.5], // tilted forward on bicycle
    accelStd: [1.2, 1.5, 1.2], // vibration vibrations but less vertical displacement than walking
    gyroMean: [0.05, 0.05, 0.1],
    gyroStd: [0.4, 0.5, 0.4], // lower rotation variance than walking
    frequency: 1.2, // pedaling rhythm (~72 rpm)
  },
  stairup: {
    accelMean: [0.8, 9.2, 1.5],
    accelStd: [2.2, 3.2, 2.0], // higher impacts
    gyroMean: [0.2, 0.4, -0.2],
    gyroStd: [1.1, 1.6, 1.2], // moderate-high rotation
    frequency: 1.4, // slightly slower cadence going up stairs
  },
  stairdown: {
    accelMean: [0.6, 9.5, 1.8],
    accelStd: [2.8, 4.2, 2.6], // highest impact peaks and std
    gyroMean: [0.3, 0.5, -0.3],
    gyroStd: [1.3, 1.8, 1.4],
    frequency: 1.6, // faster drop cadence
  },
};

// Device specific characteristics (Heterogeneity Modifiers)
interface DeviceProfile {
  accelBias: [number, number, number];
  gyroBias: [number, number, number];
  noiseMultiplier: number;
  gyroMultiplier: number; // Wrist-worn watch has much larger arm rotation sweeps
  samplingRate: number; // Hz
}

const DEVICE_PROFILES: Record<DeviceType, DeviceProfile> = {
  'Samsung S3': {
    accelBias: [-0.15, 0.1, -0.05],
    gyroBias: [0.02, -0.01, 0.01],
    noiseMultiplier: 1.1,
    gyroMultiplier: 1.0, // pocket-worn
    samplingRate: 100,
  },
  'Samsung S4': {
    accelBias: [0.02, -0.03, 0.01],
    gyroBias: [-0.005, 0.005, 0.002],
    noiseMultiplier: 0.7, // precise sensor
    gyroMultiplier: 1.0, // pocket-worn
    samplingRate: 150,
  },
  'LG Nexus 4': {
    accelBias: [0.25, -0.2, 0.15], // large sensor calibration error
    gyroBias: [-0.04, 0.03, -0.02],
    noiseMultiplier: 1.3,
    gyroMultiplier: 1.0, // pocket-worn
    samplingRate: 200, // higher sampling rate
  },
  'Galaxy Gear (Watch)': {
    accelBias: [0.05, 0.05, 0.1],
    gyroBias: [0.01, -0.02, 0.03],
    noiseMultiplier: 1.4, // watch sensor noise
    gyroMultiplier: 2.8, // wrist-worn! Arms swing and rotate MUCH more than pockets
    samplingRate: 50, // lower sampling rate
  },
};

/**
 * Generates a single feature vector for an activity and device, including noise.
 */
export function generateFeatureVector(activity: ActivityType, device: DeviceType): FeatureVector {
  const actProf = ACTIVITY_PROFILES[activity];
  const devProf = DEVICE_PROFILES[device];
  const label = ACTIVITIES.indexOf(activity);

  // Apply scaling and bias modifications
  const features: number[] = [];

  // 1. Accel Mean (applied device bias)
  const mx = actProf.accelMean[0] + devProf.accelBias[0] + (Math.random() - 0.5) * 0.1;
  const my = actProf.accelMean[1] + devProf.accelBias[1] + (Math.random() - 0.5) * 0.1;
  const mz = actProf.accelMean[2] + devProf.accelBias[2] + (Math.random() - 0.5) * 0.1;
  features.push(mx, my, mz);

  // 2. Accel Std (applied device noise multiplier)
  // For wrist-worn smartwatch, walking/biking has different motion envelope than pocket
  const isWatch = device === 'Galaxy Gear (Watch)';
  const watchAccelModifier = isWatch && (activity === 'walking' || activity === 'biking' || activity === 'stairup' || activity === 'stairdown') ? 1.4 : 1.0;

  const sx = actProf.accelStd[0] * devProf.noiseMultiplier * watchAccelModifier * (1 + (Math.random() - 0.5) * 0.15);
  const sy = actProf.accelStd[1] * devProf.noiseMultiplier * watchAccelModifier * (1 + (Math.random() - 0.5) * 0.15);
  const sz = actProf.accelStd[2] * devProf.noiseMultiplier * watchAccelModifier * (1 + (Math.random() - 0.5) * 0.15);
  features.push(sx, sy, sz);

  // 3. Gyro Mean (applied bias)
  const gmx = actProf.gyroMean[0] + devProf.gyroBias[0] + (Math.random() - 0.5) * 0.02;
  const gmy = actProf.gyroMean[1] + devProf.gyroBias[1] + (Math.random() - 0.5) * 0.02;
  const gmz = actProf.gyroMean[2] + devProf.gyroBias[2] + (Math.random() - 0.5) * 0.02;
  features.push(gmx, gmy, gmz);

  // 4. Gyro Std (applied gyroMultiplier for wrist swing!)
  const gsx = actProf.gyroStd[0] * devProf.gyroMultiplier * devProf.noiseMultiplier * (1 + (Math.random() - 0.5) * 0.15);
  const gsy = actProf.gyroStd[1] * devProf.gyroMultiplier * devProf.noiseMultiplier * (1 + (Math.random() - 0.5) * 0.15);
  const gsz = actProf.gyroStd[2] * devProf.gyroMultiplier * devProf.noiseMultiplier * (1 + (Math.random() - 0.5) * 0.15);
  features.push(gsx, gsy, gsz);

  return {
    features,
    label,
    activity,
    device,
  };
}

/**
 * Generates a full dataset.
 * Can filter to specific devices to simulate training on smartphones and testing on watches (heterogeneity test!)
 */
export function generateDataset(size: number, allowedDevices: DeviceType[] = DEVICES): FeatureVector[] {
  const dataset: FeatureVector[] = [];
  const samplesPerActivity = Math.ceil(size / ACTIVITIES.length);

  for (const activity of ACTIVITIES) {
    for (let i = 0; i < samplesPerActivity; i++) {
      // Pick a random device from the allowed list
      const device = allowedDevices[Math.floor(Math.random() * allowedDevices.length)];
      dataset.push(generateFeatureVector(activity, device));
    }
  }

  // Shuffle dataset
  return dataset.sort(() => Math.random() - 0.5).slice(0, size);
}

/**
 * Generates raw high-fidelity time series values for charts.
 * Simulates real sensor wave forms (gravity + movement oscillation + high-frequency noise + device bias).
 */
export function generateRawTimeSeries(
  activity: ActivityType,
  device: DeviceType,
  durationSeconds: number = 4,
  customSamplingRate?: number
): RawSensorSample[] {
  const actProf = ACTIVITY_PROFILES[activity];
  const devProf = DEVICE_PROFILES[device];
  const rate = customSamplingRate || devProf.samplingRate;
  const numSamples = Math.floor(durationSeconds * rate);
  const samples: RawSensorSample[] = [];

  const f = actProf.frequency;
  const isWatch = device === 'Galaxy Gear (Watch)';

  for (let i = 0; i < numSamples; i++) {
    const t = i / rate;
    const randNoise = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5; // Normal-like noise

    // 1. Accel components (Gravity + periodic gait waves + high freq noise)
    let ax = actProf.accelMean[0] + devProf.accelBias[0];
    let ay = actProf.accelMean[1] + devProf.accelBias[1];
    let az = actProf.accelMean[2] + devProf.accelBias[2];

    const noiseAmpAcc = 0.1 * devProf.noiseMultiplier;

    if (f > 0) {
      // Periodic dynamics based on activity
      const osc = Math.sin(2 * Math.PI * f * t);
      const oscHarmonic = Math.sin(4 * Math.PI * f * t) * 0.3; // add a walking/pedaling harmonic

      if (activity === 'walking') {
        ax += (osc * 1.2 + randNoise() * 0.4) * actProf.accelStd[0];
        ay += ((osc + oscHarmonic) * 1.8 + randNoise() * 0.5) * actProf.accelStd[1];
        az += (Math.cos(2 * Math.PI * f * t) * 0.9 + randNoise() * 0.3) * actProf.accelStd[2];
      } else if (activity === 'biking') {
        // biking is mostly steady with high-frequency road vibrations
        ax += (osc * 0.4 + randNoise() * 0.8) * actProf.accelStd[0];
        ay += (osc * 0.5 + randNoise() * 0.9) * actProf.accelStd[1];
        az += (randNoise() * 0.8) * actProf.accelStd[2];
      } else if (activity === 'stairup') {
        // climbing stairs has distinct larger impacts on foot striking
        const stairOsc = Math.sin(2 * Math.PI * f * t);
        const impact = stairOsc > 0.8 ? 1.5 : (stairOsc < -0.8 ? -0.8 : 0);
        ax += (stairOsc * 1.5 + randNoise() * 0.5) * actProf.accelStd[0];
        ay += ((stairOsc + impact) * 2.0 + randNoise() * 0.6) * actProf.accelStd[1];
        az += (Math.cos(2 * Math.PI * f * t) * 1.1 + randNoise() * 0.4) * actProf.accelStd[2];
      } else if (activity === 'stairdown') {
        // going down stairs has massive sharp negative acceleration spikes
        const stairOsc = Math.sin(2 * Math.PI * f * t);
        const shock = stairOsc < -0.7 ? -2.2 : 0;
        ax += (stairOsc * 1.8 + randNoise() * 0.6) * actProf.accelStd[0];
        ay += ((stairOsc * 1.5 + shock) * 2.4 + randNoise() * 0.7) * actProf.accelStd[1];
        az += (Math.cos(2 * Math.PI * f * t) * 1.4 + randNoise() * 0.5) * actProf.accelStd[2];
      }
    } else {
      // Sitting / Standing (Static sensor with tiny breathing noise or micro-sway)
      const swayPeriod = activity === 'standing' ? 0.3 : 0.05; // Standing sway
      ax += Math.sin(t * 0.5) * swayPeriod + randNoise() * noiseAmpAcc;
      ay += Math.cos(t * 0.3) * swayPeriod + randNoise() * noiseAmpAcc;
      az += Math.sin(t * 0.4) * swayPeriod + randNoise() * noiseAmpAcc;
    }

    // 2. Gyro components
    let gx = actProf.gyroMean[0] + devProf.gyroBias[0];
    let gy = actProf.gyroMean[1] + devProf.gyroBias[1];
    let gz = actProf.gyroMean[2] + devProf.gyroBias[2];

    const noiseAmpGyro = 0.05 * devProf.noiseMultiplier;

    if (f > 0) {
      const oscG = Math.cos(2 * Math.PI * f * t);
      const watchGMultiplier = isWatch ? devProf.gyroMultiplier : 1.0;

      gx += (oscG * 0.8 + randNoise() * 0.3) * actProf.gyroStd[0] * watchGMultiplier;
      gy += (Math.sin(2 * Math.PI * f * t) * 1.2 + randNoise() * 0.4) * actProf.gyroStd[1] * watchGMultiplier;
      gz += (oscG * 0.6 + randNoise() * 0.3) * actProf.gyroStd[2] * watchGMultiplier;
    } else {
      gx += randNoise() * noiseAmpGyro;
      gy += randNoise() * noiseAmpGyro;
      gz += randNoise() * noiseAmpGyro;
    }

    samples.push({
      timestamp: Math.round(t * 1000), // ms
      ax: Number(ax.toFixed(4)),
      ay: Number(ay.toFixed(4)),
      az: Number(az.toFixed(4)),
      gx: Number(gx.toFixed(4)),
      gy: Number(gy.toFixed(4)),
      gz: Number(gz.toFixed(4)),
    });
  }

  return samples;
}

/**
 * Helper to extract standard features from a raw timeseries chunk (for real physical phone connection!)
 */
export function extractFeaturesFromRawSamples(samples: RawSensorSample[]): number[] {
  if (samples.length === 0) return Array(12).fill(0);

  let sumAx = 0, sumAy = 0, sumAz = 0;
  let sumGx = 0, sumGy = 0, sumGz = 0;

  for (const s of samples) {
    sumAx += s.ax; sumAy += s.ay; sumAz += s.az;
    sumGx += s.gx; sumGy += s.gy; sumGz += s.gz;
  }

  const n = samples.length;
  const meanAx = sumAx / n;
  const meanAy = sumAy / n;
  const meanAz = sumAz / n;
  const meanGx = sumGx / n;
  const meanGy = sumGy / n;
  const meanGz = sumGz / n;

  let sqDiffAx = 0, sqDiffAy = 0, sqDiffAz = 0;
  let sqDiffGx = 0, sqDiffGy = 0, sqDiffGz = 0;

  for (const s of samples) {
    sqDiffAx += Math.pow(s.ax - meanAx, 2);
    sqDiffAy += Math.pow(s.ay - meanAy, 2);
    sqDiffAz += Math.pow(s.az - meanAz, 2);
    sqDiffGx += Math.pow(s.gx - meanGx, 2);
    sqDiffGy += Math.pow(s.gy - meanGy, 2);
    sqDiffGz += Math.pow(s.gz - meanGz, 2);
  }

  const stdAx = Math.sqrt(sqDiffAx / n);
  const stdAy = Math.sqrt(sqDiffAy / n);
  const stdAz = Math.sqrt(sqDiffAz / n);
  const stdGx = Math.sqrt(sqDiffGx / n);
  const stdGy = Math.sqrt(sqDiffGy / n);
  const stdGz = Math.sqrt(sqDiffGz / n);

  return [
    meanAx, meanAy, meanAz,
    stdAx, stdAy, stdAz,
    meanGx, meanGy, meanGz,
    stdGx, stdGy, stdGz
  ];
}
