import React from 'react';
import { useNavigate } from 'react-router-dom';
import { AudioWaveform as Waveform, Upload, BarChart3, Brain } from 'lucide-react';
import { motion } from 'framer-motion';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Upload className="w-10 h-10 text-purple-700" />,
      title: 'Upload Audio',
      description: 'Upload audio files in various formats (WAV, MP3, OGG, M4A)',
    },
    {
      icon: <Brain className="w-10 h-10 text-purple-700" />,
      title: 'Emotion Analysis',
      description: 'Our advanced neural network analyzes the emotional content of your audio',
    },
    {
      icon: <Waveform className="w-10 h-10 text-purple-700" />,
      title: 'Visualize Audio',
      description: 'See your audio represented as a waveform for detailed analysis',
    },
    {
      icon: <BarChart3 className="w-10 h-10 text-purple-700" />,
      title: 'Confidence Scores',
      description: 'Get detailed confidence scores for each detected emotion',
    },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Hero section */}
      <section className="text-center py-12 md:py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-gray-900 mb-4">
            Speech Emotion Recognition
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-8">
            Analyze the emotional content of speech using our advanced neural network model.
            Upload your audio and discover the emotions behind the words.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => navigate('/upload')}
              className="bg-purple-700 hover:bg-purple-800 text-white px-8 py-3 rounded-lg text-lg font-medium transition-colors duration-300"
            >
              Try It Now
            </button>
            <button
              onClick={() => navigate('/about')}
              className="bg-white hover:bg-gray-100 text-purple-700 border border-purple-700 px-8 py-3 rounded-lg text-lg font-medium transition-colors duration-300"
            >
              Learn More
            </button>
          </div>
        </motion.div>
      </section>

      {/* Features section */}
      <section className="py-12 md:py-20">
        <h2 className="text-3xl font-bold text-center mb-12 text-gray-900">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition-shadow duration-300"
            >
              <div className="flex justify-center mb-4">{feature.icon}</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">{feature.title}</h3>
              <p className="text-gray-600 text-center">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA section */}
      <section className="bg-purple-700 text-white rounded-2xl p-8 md:p-12 my-12 md:my-20">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to analyze your audio?</h2>
          <p className="text-lg md:text-xl mb-8 text-purple-100">
            Upload your audio file and get instant emotion analysis with our state-of-the-art neural network.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-white text-purple-700 hover:bg-purple-100 px-8 py-3 rounded-lg text-lg font-medium transition-colors duration-300"
          >
            Start Analyzing
          </button>
        </div>
      </section>
    </div>
  );
};

export default HomePage;