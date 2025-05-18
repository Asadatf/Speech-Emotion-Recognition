import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Play, Pause } from "lucide-react";

interface AudioWaveformProps {
  audioUrl: string;
}

const AudioWaveform: React.FC<AudioWaveformProps> = ({ audioUrl }) => {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurfer = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  // Initialize WaveSurfer when component mounts
  useEffect(() => {
    if (!waveformRef.current) return;

    // Create WaveSurfer instance
    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: "#9333ea",
      progressColor: "#6D28D9",
      cursorColor: "#4c1d95",
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      cursorWidth: 1,
      height: 100,
    });

    // Set up event listeners
    ws.on("ready", () => {
      setDuration(ws.getDuration());
    });

    ws.on("audioprocess", () => {
      setCurrentTime(ws.getCurrentTime());
    });

    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    // Store WaveSurfer instance
    wavesurfer.current = ws;

    // Cleanup function
    return () => {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
        wavesurfer.current = null;
      }
    };
  }, []);

  // Handle audio URL changes
  useEffect(() => {
    const loadAudio = async () => {
      if (wavesurfer.current && audioUrl) {
        try {
          await wavesurfer.current.load(audioUrl);
        } catch (error) {
          console.error("Error loading audio:", error);
        }
      }
    };

    loadAudio();
  }, [audioUrl]);

  const togglePlayPause = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
    }
  };

  // Format time in mm:ss
  const formatTime = (time: number) => {
    if (!isFinite(time) || time < 0) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  return (
    <div className="audio-waveform">
      <div ref={waveformRef} className="mb-4"></div>

      <div className="flex items-center justify-between">
        <button
          onClick={togglePlayPause}
          className="bg-purple-700 hover:bg-purple-800 text-white p-3 rounded-full transition-colors duration-300"
        >
          {isPlaying ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5" />
          )}
        </button>

        <div className="text-sm text-gray-600">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>
    </div>
  );
};

export default AudioWaveform;
