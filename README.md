# Speech Emotion Recognition Web Application

A full-stack web application that analyzes emotions in speech using a CNN + HuBERT model trained on the RAVDESS dataset.

![EmotionSense Screenshot](https://images.pexels.com/photos/3756879/pexels-photo-3756879.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

## Features

- **Audio Upload**: Support for multiple audio formats (WAV, MP3, OGG, M4A)
- **Real-time Waveform Visualization**: Interactive audio waveform display with playback controls
- **Emotion Analysis**: Detects 7 different emotions:
  - Happy
  - Sad
  - Angry
  - Neutral
  - Fearful
  - Disgusted
  - Surprised
- **Results Export**: Download analysis results as PDF
- **Share Functionality**: Share results via Web Share API or clipboard
- **Analysis History**: Track and review past emotion analyses
- **Responsive Design**: Fully responsive UI that works on all devices

## Tech Stack

### Frontend

- React 18
- TypeScript
- Tailwind CSS
- Framer Motion for animations
- WaveSurfer.js for audio visualization
- Lucide React for icons
- jsPDF for PDF generation

### Backend

- Node.js with Express
- CNN + HuBERT model trained on RAVDESS dataset
- Multer for file uploads
- CORS for cross-origin resource sharing

## Getting Started

1. Clone the repository
2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

4. Start the backend server:

```bash
npm run server
```

The application will be available at `http://localhost:5173`

## Model Architecture

Our emotion recognition system combines a Convolutional Neural Network (CNN) with HuBERT (Hidden-Unit BERT) pre-trained representations. The model was trained on the RAVDESS dataset, which contains voice recordings from professional actors expressing various emotions.

### Model Features

- Uses HuBERT for robust speech representations
- CNN layers for emotion-specific feature extraction
- Trained on the RAVDESS dataset for high accuracy
- Supports multiple languages and accents
- Real-time processing capabilities

## API Endpoints

### POST `/api/upload`

Upload and analyze audio file

- Method: POST
- Content-Type: multipart/form-data
- Body: audio file
- Returns: Emotion analysis results

### GET `/api/history`

Retrieve analysis history

- Method: GET
- Returns: Array of past analyses

## Environment Setup

The application requires the following environment variables:

```env
VITE_API_URL=http://localhost:3000
```

## Acknowledgments

- RAVDESS Dataset for providing high-quality emotional speech data
- HuBERT model for advanced speech representations
- Meta AI Research for the original HuBERT implementation
