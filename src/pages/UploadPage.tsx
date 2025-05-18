import React, { useState, useRef, useEffect } from "react";
import { Upload, Play, Pause, AlertCircle, Check } from "lucide-react";
import { motion } from "framer-motion";
import AudioWaveform from "../components/AudioWaveform";
import EmotionResults from "../components/EmotionResults";
import axios from "axios";

interface UploadState {
  status: "idle" | "uploading" | "processing" | "success" | "error";
  progress: number;
  errorMessage?: string;
}

interface PredictionResult {
  primaryEmotion: string;
  confidenceScores: Record<string, number>;
  audioUrl: string;
  timestamp: string;
}

const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>({
    status: "idle",
    progress: 0,
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<number | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);

    if (selectedFile) {
      setAudioUrl(URL.createObjectURL(selectedFile));
      setUploadState({ status: "idle", progress: 0 });
      setPrediction(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];

    if (droppedFile && droppedFile.type.startsWith("audio/")) {
      setFile(droppedFile);
      setAudioUrl(URL.createObjectURL(droppedFile));
      setUploadState({ status: "idle", progress: 0 });
      setPrediction(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  // Clear any intervals on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const handleUpload = async () => {
    if (!file) return;

    setUploadState({ status: "uploading", progress: 0 });

    // Clear any existing interval
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }

    const formData = new FormData();
    formData.append("audio", file);

    // Set up progress simulation that goes to 85% during upload
    progressIntervalRef.current = setInterval(() => {
      setUploadState((prev) => {
        // Only increase progress if we're still in uploading or processing state
        if (
          (prev.status === "uploading" || prev.status === "processing") &&
          prev.progress < 85
        ) {
          return { ...prev, progress: prev.progress + 5 };
        }
        return prev;
      });
    }, 500);

    try {
      // Set timeout for the request to prevent indefinite waiting
      const response = await axios.post(
        "http://localhost:3000/api/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 180000, // 3 minutes timeout - model processing can take time
          withCredentials: false,
        }
      );

      // Show final processing state before success
      setUploadState({ status: "processing", progress: 95 });

      // Clear the progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      // Short delay before showing success to make transition smoother
      setTimeout(() => {
        setUploadState({ status: "success", progress: 100 });
        setPrediction(response.data);
      }, 500);
    } catch (error) {
      console.error("Upload error:", error);

      // Clear the progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      setUploadState({
        status: "error",
        progress: 0,
        errorMessage:
          error instanceof Error
            ? error.message
            : "Failed to upload audio. Please try again.",
      });
    }
  };

  // Clean up audio URL on component unmount
  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-6 text-center">
          Upload Audio for Emotion Analysis
        </h1>

        {/* Upload area */}
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center mb-8 transition-colors duration-300 ${
            uploadState.status === "error"
              ? "border-red-500 bg-red-50"
              : "border-purple-300 hover:border-purple-500 bg-purple-50"
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          {!file ? (
            <>
              <Upload className="w-16 h-16 mx-auto text-purple-600 mb-4" />
              <h2 className="text-xl font-semibold mb-2">
                Drag & drop your audio file here
              </h2>
              <p className="text-gray-600 mb-4">
                Supports WAV, MP3, OGG, and M4A formats
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-purple-700 hover:bg-purple-800 text-white px-6 py-2 rounded-lg transition-colors duration-300"
              >
                Browse Files
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="audio/*"
                className="hidden"
              />
            </>
          ) : (
            <>
              <div className="mb-4">
                <div className="bg-white p-4 rounded-lg shadow-sm inline-block">
                  <div className="flex items-center">
                    <div className="mr-3 bg-purple-100 p-2 rounded-full">
                      <Play className="w-5 h-5 text-purple-700" />
                    </div>
                    <div className="text-left">
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {uploadState.status === "error" && (
                <div className="flex items-center justify-center text-red-600 mb-4">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  <span>{uploadState.errorMessage || "An error occurred"}</span>
                </div>
              )}

              <div className="flex flex-wrap justify-center gap-3">
                <button
                  onClick={() => {
                    setFile(null);
                    setAudioUrl(null);
                    setPrediction(null);
                    setUploadState({ status: "idle", progress: 0 });
                  }}
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-6 py-2 rounded-lg transition-colors duration-300"
                >
                  Remove
                </button>

                {uploadState.status !== "success" && (
                  <button
                    onClick={handleUpload}
                    disabled={
                      uploadState.status === "uploading" ||
                      uploadState.status === "processing"
                    }
                    className={`bg-purple-700 hover:bg-purple-800 text-white px-6 py-2 rounded-lg transition-colors duration-300 ${
                      uploadState.status === "uploading" ||
                      uploadState.status === "processing"
                        ? "opacity-70 cursor-not-allowed"
                        : ""
                    }`}
                  >
                    {uploadState.status === "uploading"
                      ? "Uploading..."
                      : uploadState.status === "processing"
                      ? "Processing..."
                      : "Analyze Audio"}
                  </button>
                )}
              </div>

              {(uploadState.status === "uploading" ||
                uploadState.status === "processing") && (
                <div className="mt-4">
                  <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-purple-600 transition-all duration-300 ease-out"
                      style={{ width: `${uploadState.progress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    {uploadState.status === "uploading"
                      ? "Uploading..."
                      : "Processing using ensemble model..."}{" "}
                    {uploadState.progress}%
                    {uploadState.progress >= 85 &&
                      uploadState.status === "processing" && (
                        <span className="ml-2 italic text-xs">
                          (this may take 1-2 minutes as the model analyzes your
                          audio)
                        </span>
                      )}
                  </p>
                </div>
              )}

              {uploadState.status === "success" && (
                <div className="flex items-center justify-center text-green-600 mt-4">
                  <Check className="w-5 h-5 mr-2" />
                  <span>Analysis complete!</span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Audio waveform and playback controls */}
        {audioUrl && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white p-6 rounded-xl shadow-md mb-8"
          >
            <h2 className="text-xl font-semibold mb-4">Audio Waveform</h2>
            <AudioWaveform audioUrl={audioUrl} />
          </motion.div>
        )}

        {/* Emotion analysis results */}
        {prediction && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <EmotionResults prediction={prediction} />
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default UploadPage;
