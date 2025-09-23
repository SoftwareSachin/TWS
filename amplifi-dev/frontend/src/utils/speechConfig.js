import * as SpeechSDK from "microsoft-cognitiveservices-speech-sdk";

const speechConfig = SpeechSDK.SpeechConfig.fromSubscription(
  process.env.NEXT_PUBLIC_SPEECH_KEY || "a",
  "southeastasia",
);
speechConfig.speechRecognitionLanguage = "en-US";
speechConfig.speechSynthesisVoiceName = "en-US-NovaTurboMultilingualNeural"; // You can change the voice here

const recognizerConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
const recognizer = () =>
  new SpeechSDK.SpeechRecognizer(speechConfig, recognizerConfig);

const audioPlayer = () => new SpeechSDK.SpeakerAudioDestination();

const synthesizerConfig = () =>
  SpeechSDK.AudioConfig.fromSpeakerOutput(audioPlayer());
const synthesizer = () =>
  new SpeechSDK.SpeechSynthesizer(speechConfig, synthesizerConfig());

export { recognizer, recognizerConfig, synthesizer, audioPlayer, speechConfig };
