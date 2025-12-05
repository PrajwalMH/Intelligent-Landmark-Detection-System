import React, { useState, useEffect } from "react";
import axios from "axios";


export default function UploadForm({ onResults }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    try {
      const res = await axios.post("/api/predict", formData);
      onResults(file, res.data);
    } catch (err) {
      console.error("Upload failed", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card p-4 shadow-sm">
      <h5 className="mb-3">Upload an Image</h5>
      <input
        type="file"
        className="form-control mb-3"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button className="btn btn-primary" onClick={handleUpload} disabled={loading}>
        {loading ? "Processing..." : "Submit"}
      </button>
    </div>
  );
}
