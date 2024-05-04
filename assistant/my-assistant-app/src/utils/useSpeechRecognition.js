export const startSpeechRecognition = () => {
    return new Promise((resolve, reject) => {
      const recognition = new window.SpeechRecognition();
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        resolve(transcript);
      };
      recognition.onerror = (event) => {
        reject(event.error);
      };
      recognition.start();
      return recognition;
    });
  };
  
  export const stopSpeechRecognition = (recognition) => {
    recognition.stop();
  };
  
  export const speakResponse = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };