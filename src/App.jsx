import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import { detect, detectVideo } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [model, setModel] = useState({ net: null, inputShape: [1, 0, 0, 3] });
  const [selectedModel, setSelectedModel] = useState("YOLO_8_N_Ori");

  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    tf.ready().then(async () => {
      const modelUrl = `${window.location.href}/${selectedModel}_web_model/model.json`;
      const yolov8 = await tf.loadGraphModel(modelUrl, {
        onProgress: (fractions) => {
          setLoading({ loading: true, progress: fractions });
        },
      });

      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      const warmupResults = yolov8.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
      });

      tf.dispose([warmupResults, dummyInput]);
    });
  }, [selectedModel]);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  return (
    <div className="App">
      <div className="model-selector">
        <label htmlFor="model-select">Select Model:</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={handleModelChange}
          className="modelSelection"
        >
          <option value="YOLO_8_N_Ori">YOLO_8_N_Ori</option>
          <option value="YOLO_8_M_Ori">YOLO_8_M_Ori</option>
        </select>
      </div>
      {loading.loading && (
        <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>
      )}
      <div className="header">
        <h1>ESD Suit Detection</h1>
        <p>Demo#1</p>
        <p>
          Serving : <code className="code">{selectedModel}</code>
        </p>
      </div>
      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad={() => detect(imageRef.current, model, canvasRef.current)}
        />
        <video
          autoPlay
          muted
          ref={cameraRef}
          onPlay={() =>
            detectVideo(cameraRef.current, model, canvasRef.current)
          }
        />
        <video
          autoPlay
          muted
          ref={videoRef}
          onPlay={() => detectVideo(videoRef.current, model, canvasRef.current)}
        />
        <canvas
          width={model.inputShape[1]}
          height={model.inputShape[2]}
          ref={canvasRef}
        />
      </div>
      <ButtonHandler
        imageRef={imageRef}
        cameraRef={cameraRef}
        videoRef={videoRef}
      />
    </div>
  );
};

export default App;
