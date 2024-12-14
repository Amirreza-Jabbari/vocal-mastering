
---

# Vocal Mastering Application

## Overview
The Vocal Mastering Application is a Django-based web application designed to facilitate the uploading and processing of vocal audio files. It utilizes advanced audio processing techniques to enhance the quality of vocal recordings, making it suitable for musicians, producers, and audio engineers.

## Features
- **File Upload:** Users can upload vocal audio files in various formats (WAV, MP3, M4A).
- **Asynchronous Processing:** Audio processing is handled in the background using Celery, allowing users to continue using the application while their files are being processed.
- **Advanced Audio Processing:** The application employs a range of audio processing techniques, including noise reduction, dynamic compression, equalization, and loudness normalization.
- **Job Status Tracking:** Users can check the status of their audio processing jobs and download the mastered audio once completed.

## Technologies Used
- **Django:** A high-level Python web framework for building web applications.
- **Celery:** An asynchronous task queue/job queue based on distributed message passing.
- **Librosa:** A Python package for music and audio analysis.
- **NumPy:** A library for numerical computations in Python.
- **SoundFile:** A library for reading and writing sound files.
- **Pyloudnorm:** A library for loudness normalization.

## Installation
To set up the Vocal Mastering Application locally, follow these steps:

### Clone the Repository:
```bash
git clone https://github.com/Amirreza-Jabbari/vocal-mastering.git
cd vocal-mastering
```

### Create a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Set Up the Database:
```bash
python manage.py migrate
```

### Run the Development Server:
```bash
python manage.py runserver
```

### Access the Application:
Open your web browser and navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Usage
1. **Upload Vocal:** Navigate to the upload page and select an audio file to upload.
2. **Processing:** After uploading, the application will process the audio file in the background. You will be redirected to a job status page.
3. **Download Mastered Audio:** Once processing is complete, you can download the mastered audio file.

## Code Structure
The application is organized into several key components:
- `forms.py`: Contains the form for uploading audio files.
- `models.py`: Defines the data models for storing audio files and processing jobs.
- `processors.py`: Implements the audio processing logic.
- `tasks.py`: Contains the Celery tasks for asynchronous processing.
- `views.py`: Handles the web requests and responses.
- `urls.py`: Defines the URL routing for the application.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## result Screenshot
[![result.png](https://i.postimg.cc/3NDJPFcs/result.png)](https://postimg.cc/sGsrW7ym)

---
