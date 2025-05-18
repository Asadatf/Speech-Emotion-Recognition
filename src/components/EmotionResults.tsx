import React from "react";
import { motion } from "framer-motion";
import jsPDF from "jspdf";

interface PredictionResult {
  primaryEmotion: string;
  confidenceScores: Record<string, number>;
  audioUrl: string;
  timestamp: string;
}

interface EmotionResultsProps {
  prediction: PredictionResult;
}

const EmotionResults: React.FC<EmotionResultsProps> = ({ prediction }) => {
  // Colors for different emotions
  const emotionColors: Record<
    string,
    { bg: string; text: string; light: string }
  > = {
    Happy: {
      bg: "bg-yellow-500",
      text: "text-yellow-600",
      light: "bg-yellow-100",
    },
    Sad: { bg: "bg-blue-500", text: "text-blue-600", light: "bg-blue-100" },
    Angry: { bg: "bg-red-500", text: "text-red-600", light: "bg-red-100" },
    Neutral: { bg: "bg-gray-500", text: "text-gray-600", light: "bg-gray-100" },
    Fearful: {
      bg: "bg-indigo-500",
      text: "text-indigo-600",
      light: "bg-indigo-100",
    },
    Disgusted: {
      bg: "bg-green-500",
      text: "text-green-600",
      light: "bg-green-100",
    },
    Surprised: {
      bg: "bg-orange-500",
      text: "text-orange-600",
      light: "bg-orange-100",
    },
  };

  // Format percentage from decimal
  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(1) + "%";
  };

  // Sort emotions by confidence score (highest first)
  const sortedEmotions = Object.entries(prediction.confidenceScores).sort(
    (a, b) => b[1] - a[1]
  );

  // Handle download results as PDF
  const handleDownload = () => {
    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    let yPosition = 20;

    // Add title
    pdf.setFontSize(20);
    pdf.setTextColor(85, 26, 139); // Purple color
    pdf.text("Speech Emotion Analysis Results", pageWidth / 2, yPosition, {
      align: "center",
    });

    // Add timestamp
    yPosition += 20;
    pdf.setFontSize(12);
    pdf.setTextColor(100, 100, 100);
    pdf.text(
      `Analysis Date: ${new Date(prediction.timestamp).toLocaleString()}`,
      20,
      yPosition
    );

    // Add primary emotion
    yPosition += 20;
    pdf.setFontSize(16);
    pdf.setTextColor(0, 0, 0);
    pdf.text("Primary Emotion Detected:", 20, yPosition);
    yPosition += 10;
    pdf.setFontSize(14);
    pdf.setTextColor(85, 26, 139);
    pdf.text(prediction.primaryEmotion, 20, yPosition);

    // Add confidence scores
    yPosition += 20;
    pdf.setFontSize(16);
    pdf.setTextColor(0, 0, 0);
    pdf.text("Confidence Scores:", 20, yPosition);

    // Add each emotion score
    yPosition += 10;
    pdf.setFontSize(12);
    sortedEmotions.forEach(([emotion, score]) => {
      yPosition += 10;
      pdf.text(`${emotion}: ${formatPercentage(score)}`, 20, yPosition);
    });

    // Add audio file reference
    yPosition += 20;
    pdf.setFontSize(12);
    pdf.setTextColor(100, 100, 100);
    pdf.text(
      `Audio File: ${prediction.audioUrl.split("/").pop()}`,
      20,
      yPosition
    );

    // Save the PDF
    pdf.save(`emotion-analysis-${new Date().toISOString()}.pdf`);
  };

  // Handle share results
  const handleShare = async () => {
    const shareData = {
      title: "Speech Emotion Analysis Results",
      text: `Primary emotion detected: ${prediction.primaryEmotion}\nAnalysis timestamp: ${prediction.timestamp}`,
      url: window.location.href,
    };

    try {
      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        // Fallback to clipboard copy if Web Share API is not available
        await navigator.clipboard.writeText(
          `Speech Emotion Analysis Results\n\n` +
            `Primary emotion: ${prediction.primaryEmotion}\n` +
            `Confidence scores:\n${Object.entries(prediction.confidenceScores)
              .map(
                ([emotion, score]) => `${emotion}: ${formatPercentage(score)}`
              )
              .join("\n")}\n\nAnalyzed on: ${prediction.timestamp}`
        );
        alert("Results copied to clipboard!");
      }
    } catch (error) {
      console.error("Error sharing results:", error);
      alert("Failed to share results. Please try again.");
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-md">
      <h2 className="text-2xl font-bold mb-6 text-center">
        Emotion Analysis Results
      </h2>

      {/* Primary emotion */}
      <div className="mb-8 text-center">
        <p className="text-lg text-gray-700 mb-2">Primary Emotion Detected</p>
        <div className="inline-block">
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5 }}
            className={`text-3xl font-bold py-2 px-6 rounded-lg ${
              emotionColors[prediction.primaryEmotion]?.light || "bg-purple-100"
            } ${
              emotionColors[prediction.primaryEmotion]?.text ||
              "text-purple-600"
            }`}
          >
            {prediction.primaryEmotion}
          </motion.div>
        </div>
      </div>

      {/* Confidence scores */}
      <div>
        <h3 className="text-xl font-semibold mb-4">Confidence Scores</h3>
        <div className="space-y-4">
          {sortedEmotions.map(([emotion, score], index) => (
            <motion.div
              key={emotion}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div className="flex justify-between mb-1">
                <span className="text-gray-700">{emotion}</span>
                <span className="font-medium">{formatPercentage(score)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${score * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.2 + index * 0.1 }}
                  className={`h-full ${
                    emotionColors[emotion]?.bg || "bg-purple-600"
                  }`}
                ></motion.div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="mt-8 flex justify-center space-x-4">
        <button
          onClick={handleDownload}
          className="bg-purple-700 hover:bg-purple-800 text-white px-4 py-2 rounded-lg transition-colors duration-300"
        >
          Download Results
        </button>
        <button
          onClick={handleShare}
          className="bg-white border border-purple-700 text-purple-700 hover:bg-purple-50 px-4 py-2 rounded-lg transition-colors duration-300"
        >
          Share Results
        </button>
      </div>
    </div>
  );
};

export default EmotionResults;
