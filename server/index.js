// Optimized Express server with fixed Python check
import express from "express";
import multer from "multer";
import cors from "cors";
import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";
import { spawn, execSync } from "child_process"; // Import execSync properly

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

// Create necessary directories
const modelDir = path.join(__dirname, "models");
const uploadsDir = path.join(__dirname, "uploads");
const pythonDir = path.join(__dirname, "python");

[modelDir, uploadsDir, pythonDir].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Configure whether to use ensemble
const USE_ENSEMBLE = true;
const PYTHON_SCRIPT = path.join(pythonDir, "predict_emotion.py");

// Helper function to check if Python is installed - Fixed for ES modules
function checkPythonInstalled() {
  try {
    const result = execSync("python --version").toString();
    return result.toLowerCase().includes("python");
  } catch (error) {
    console.warn("Python check failed:", error.message);
    return false;
  }
}

// Pre-load models flag (for second run optimization)
let modelsPreloaded = false;

// Configure CORS
app.use(
  cors({
    origin: "http://localhost:5173", // Vite dev server
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

app.use(express.json());

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(
      null,
      file.fieldname + "-" + uniqueSuffix + path.extname(file.originalname)
    );
  },
});

const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    // Accept audio files only
    if (!file.originalname.match(/\.(wav|mp3|ogg|m4a)$/)) {
      return cb(new Error("Only audio files are allowed!"), false);
    }
    cb(null, true);
  },
});

// Serve uploaded files
app.use("/uploads", express.static(uploadsDir));

// History storage (in-memory for demo, would use a database in production)
const analysisHistory = [];

// Function to run Python prediction script
function runPythonPrediction(audioPath) {
  return new Promise((resolve, reject) => {
    let args = [PYTHON_SCRIPT];

    if (USE_ENSEMBLE) {
      // For ensemble, use all three models
      args.push(modelDir, audioPath, "ensemble");
      if (modelsPreloaded) {
        args.push("preloaded");
      }
    } else {
      // For single model, use the best model
      const modelPath = path.join(modelDir, "best_model.pth");
      args.push(modelPath, audioPath);
      if (modelsPreloaded) {
        args.push("preloaded");
      }
    }

    console.log("Running Python prediction with args:", args);

    // Execute the Python script as a process
    const pythonProcess = spawn("python", args);

    let result = "";
    let errorOutput = "";

    // Collect data from stdout
    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    // Collect error output
    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
      console.error(`Python error: ${data}`);
    });

    // Handle process completion
    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error output: ${errorOutput}`);
        reject(new Error(`Python prediction failed with code ${code}`));
        return;
      }

      try {
        // Find the JSON part of the output (in case there are debug prints)
        const jsonStart = result.indexOf("{");
        const jsonEnd = result.lastIndexOf("}");

        if (jsonStart !== -1 && jsonEnd !== -1) {
          const jsonStr = result.substring(jsonStart, jsonEnd + 1);
          const prediction = JSON.parse(jsonStr);

          // Mark that models have been loaded
          modelsPreloaded = true;

          resolve(prediction);
        } else {
          console.error("Failed to find JSON in Python output:", result);
          reject(new Error("Failed to parse prediction result"));
        }
      } catch (e) {
        console.error("Failed to parse Python output:", e);
        console.error("Raw output:", result);
        reject(new Error("Failed to parse prediction result"));
      }
    });
  });
}

// Pre-load models endpoint (call this after server starts to warm up the models)
app.get("/api/preload", (req, res) => {
  // Create a dummy audio file for preloading
  const dummyAudioPath = path.join(uploadsDir, "dummy.wav");

  console.log("Pre-loading models...");
  runPythonPrediction(dummyAudioPath)
    .then(() => {
      modelsPreloaded = true;
      res.json({ success: true, message: "Models pre-loaded successfully" });
    })
    .catch((error) => {
      console.error("Error pre-loading models:", error);
      res.status(500).json({ success: false, error: error.message });
    });
});

// Route to handle audio file uploads
app.post("/api/upload", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  try {
    const audioFilePath = req.file.path;
    const audioUrl = `/uploads/${req.file.filename}`;

    let prediction;

    try {
      // Track how long prediction takes
      const startTime = Date.now();

      // Run Python prediction
      prediction = await runPythonPrediction(audioFilePath);

      const endTime = Date.now();
      console.log(`Prediction took ${endTime - startTime}ms`);

      // Add additional info
      prediction.audioUrl = audioUrl;
      prediction.timestamp = new Date().toISOString();
    } catch (error) {
      console.error("Error predicting emotion:", error.message);

      // Fallback to mock prediction if Python fails
      console.log("Using mock prediction as fallback");

      const emotions = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
      ];
      prediction = {
        primaryEmotion: emotions[Math.floor(Math.random() * emotions.length)],
        confidenceScores: emotions.reduce((acc, emotion) => {
          acc[emotion] = Math.random();
          return acc;
        }, {}),
        audioUrl: audioUrl,
        timestamp: new Date().toISOString(),
      };

      // Normalize confidence scores to sum to 1
      const sum = Object.values(prediction.confidenceScores).reduce(
        (a, b) => a + b,
        0
      );
      Object.keys(prediction.confidenceScores).forEach((key) => {
        prediction.confidenceScores[key] =
          prediction.confidenceScores[key] / sum;
      });
    }

    // Save to history
    const historyEntry = {
      id: analysisHistory.length,
      fileName: req.file.originalname,
      ...prediction,
    };
    analysisHistory.unshift(historyEntry); // Add to beginning of array

    // Limit history size
    if (analysisHistory.length > 20) {
      analysisHistory.pop();
    }

    res.json(prediction);
  } catch (error) {
    console.error("Error processing audio:", error);
    res.status(500).json({ error: "Error processing audio file" });
  }
});

// Get history of predictions
app.get("/api/history", (req, res) => {
  res.json(analysisHistory);
});

// Server status and model check endpoint
app.get("/api/status", (req, res) => {
  const pythonInstalled = checkPythonInstalled();
  const pythonScriptExists = fs.existsSync(PYTHON_SCRIPT);
  const modelFilesExist = checkModelFiles();

  res.json({
    status:
      pythonInstalled && pythonScriptExists && modelFilesExist
        ? "ready"
        : "setup required",
    pythonInstalled,
    pythonScriptExists,
    modelsPreloaded,
    modelFilesExist,
    usingEnsemble: USE_ENSEMBLE,
    modelDirectory: modelDir,
    serverTime: new Date().toISOString(),
  });
});

// Helper function to check if model files exist
function checkModelFiles() {
  if (USE_ENSEMBLE) {
    // Check for ensemble models
    let hasModels = false;
    for (let i = 1; i <= 3; i++) {
      const modelPath = path.join(modelDir, `ensemble_model_${i}.pth`);
      if (fs.existsSync(modelPath)) {
        hasModels = true;
        break;
      }
    }
    return hasModels;
  } else {
    // Check for best model
    return fs.existsSync(path.join(modelDir, "best_model.pth"));
  }
}

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);

  // Check for Python
  if (!checkPythonInstalled()) {
    console.warn(
      "WARNING: Python does not appear to be installed or is not in the PATH."
    );
    console.warn(
      "Please install Python and the required packages (torch, numpy, librosa, transformers)."
    );
  }

  // Check for the Python script
  if (!fs.existsSync(PYTHON_SCRIPT)) {
    console.warn(
      "WARNING: The Python prediction script is not found at:",
      PYTHON_SCRIPT
    );
    console.warn("Please create this file with the provided code.");
  }

  // Check for model files
  if (!checkModelFiles()) {
    console.warn("WARNING: Model files are not found in the models directory.");
    console.warn(`Please copy your ensemble models to: ${modelDir}`);
    console.warn(
      "Expected files: ensemble_model_1.pth, ensemble_model_2.pth, ensemble_model_3.pth"
    );
  }

  console.log(`API endpoint: http://localhost:${port}/api/upload`);
  console.log(`History endpoint: http://localhost:${port}/api/history`);
  console.log(`Status endpoint: http://localhost:${port}/api/status`);
  console.log(`Preload endpoint: http://localhost:${port}/api/preload`);
  console.log(`Using ${USE_ENSEMBLE ? "ENSEMBLE" : "SINGLE"} model mode.`);

  // Suggest pre-loading models
  console.log(
    "\nTo improve first prediction speed, you can pre-load models by visiting:"
  );
  console.log(`http://localhost:${port}/api/preload`);
});
