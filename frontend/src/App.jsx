// frontend/src/App.jsx
import React, { useState } from "react";
import { uploadImage } from "./api";
import "./App.css";

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [bbox, setBbox] = useState(null);
  const [label, setLabel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImageFile(file);
    setImageUrl(URL.createObjectURL(file));
    setBbox(null);
    setLabel(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!imageFile) return;
    try {
      setLoading(true);
      setError(null);

      const result = await uploadImage(imageFile);
      // result.bbox is normalized [0,1]
      setBbox(result.bbox);
      setLabel(result.label);
    } catch (err) {
      console.error(err);
      setError("Failed to get prediction from server.");
    } finally {
      setLoading(false);
    }
  };

  let bboxStyle = {};
  if (bbox && imageUrl) {
    const [x1, y1, x2, y2] = bbox;
    bboxStyle = {
      left: `${x1 * 100}%`,
      top: `${y1 * 100}%`,
      width: `${(x2 - x1) * 100}%`,
      height: `${(y2 - y1) * 100}%`,
    };
  }

  return (
    <div className="app">
      <h1>Landmark Detector</h1>

      <div className="controls">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleSubmit} disabled={!imageFile || loading}>
          {loading ? "Detecting..." : "Detect Landmark"}
        </button>
        {error && <p className="error">{error}</p>}
      </div>

      {imageUrl && (
        <div className="image-wrapper">
          <div className="image-container">
            <img src={imageUrl} alt="upload" className="image" />
            {bbox && (
              <div className="bbox">
                <div className="bbox-inner" style={bboxStyle} />
              </div>
            )}
          </div>
          {label && <p className="label">Prediction: {label}</p>}
        </div>
      )}
    </div>
  );
}

export default App;
