import React from 'react';
import { Code, BookOpen, Database, Heart } from 'lucide-react';
import { motion } from 'framer-motion';

const AboutPage: React.FC = () => {
  const sections = [
    {
      icon: <Code className="w-8 h-8 text-purple-700" />,
      title: 'The Technology',
      content: `Our speech emotion recognition system uses a deep neural network trained on the popular Kaggle speech emotion recognition datasets. The model processes audio features such as MFCCs (Mel-frequency cepstral coefficients), chroma, mel spectrograms, and more to identify emotional patterns in speech.`
    },
    {
      icon: <BookOpen className="w-8 h-8 text-purple-700" />,
      title: 'The Research',
      content: `Speech emotion recognition is a growing field at the intersection of audio signal processing and machine learning. Our approach builds on research from leading publications in affective computing and speech analysis, with a focus on cross-cultural applicability and robustness to environmental noise.`
    },
    {
      icon: <Database className="w-8 h-8 text-purple-700" />,
      title: 'The Data',
      content: `We've trained our model on multiple emotional speech datasets, including RAVDESS, TESS, SAVEE, and CREMA-D. These collectively provide a diverse range of speakers, emotions, and linguistic contexts to ensure our model generalizes well to real-world scenarios.`
    },
    {
      icon: <Heart className="w-8 h-8 text-purple-700" />,
      title: 'Applications',
      content: `Our technology can be applied in various domains, including mental health monitoring, customer service quality assessment, voice assistant personalization, media content analysis, and automotive safety through driver emotional state monitoring.`
    }
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <h1 className="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-8">About Our Speech Emotion Recognition</h1>
        
        <div className="bg-white p-6 md:p-8 rounded-xl shadow-md mb-8">
          <h2 className="text-2xl font-bold mb-6 text-gray-900">How It Works</h2>
          <p className="text-gray-700 mb-4">
            Our speech emotion recognition system analyzes audio recordings to detect emotions such as happiness, sadness, anger, fear, disgust, surprise, and neutral states. This technology leverages advanced deep learning algorithms trained on diverse datasets to recognize subtle emotional cues in speech patterns.
          </p>
          <p className="text-gray-700 mb-4">
            The process begins by extracting acoustic features from the audio signal, including pitch, energy, tempo, and spectral characteristics. These features are then processed through our neural network model, which has been trained to identify emotional patterns in speech.
          </p>
          <p className="text-gray-700">
            The model outputs confidence scores for each possible emotion, indicating how likely each emotion is present in the speech. The emotion with the highest confidence score is presented as the primary detected emotion.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {sections.map((section, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white p-6 rounded-xl shadow-md"
            >
              <div className="flex items-center mb-4">
                <div className="bg-purple-100 p-3 rounded-full mr-4">
                  {section.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900">{section.title}</h3>
              </div>
              <p className="text-gray-700">{section.content}</p>
            </motion.div>
          ))}
        </div>
        
        <div className="bg-purple-700 text-white rounded-xl p-8 text-center">
          <h2 className="text-2xl font-bold mb-4">Ready to experience it yourself?</h2>
          <p className="text-lg mb-6">
            Upload your audio file and see our speech emotion recognition in action.
          </p>
          <a 
            href="/upload" 
            className="inline-block bg-white text-purple-700 hover:bg-purple-100 px-6 py-3 rounded-lg font-medium transition-colors duration-300"
          >
            Try It Now
          </a>
        </div>
      </motion.div>
    </div>
  );
};

export default AboutPage;