import React, { useRef, useEffect } from "react";

const ResultViewer = ({ imageUrl, custom, faster_rcnn }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");

      // Resize canvas to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Helper to clamp values within image bounds
      const clamp = (val, min, max) => Math.max(min, Math.min(val, max));

      // Draw a bounding box
      const drawBox = (box, label, color) => {
        const [x1, y1, x2, y2] = box;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = color;
        ctx.font = "16px Arial";
        ctx.fillText(label, x1 + 4, y1 - 6);
      };

      // Draw Faster R-CNN box if score is high enough
      if (faster_rcnn?.scores?.[0] > 0.3) {
        drawBox(faster_rcnn.boxes[0], `FasterRCNN: ${faster_rcnn.labels[0]}`, "red");
      }

      // Draw Custom box if score is high enough
      if (custom?.scores?.[0] > 0.3) {
        const [x1, y1, x2, y2] = custom.boxes[0];
        const scaledBox = [
          clamp(x1 * img.width, 0, img.width),
          clamp(y1 * img.height, 0, img.height),
          clamp(x2 * img.width, 0, img.width),
          clamp(y2 * img.height, 0, img.height),
        ];
        drawBox(scaledBox, `Custom: ${custom.labels[0]}`, "blue");
      }
    };
  }, [imageUrl, custom, faster_rcnn]);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <canvas ref={canvasRef} style={{ border: "1px solid #ccc" }} />
    </div>
  );
};

export default ResultViewer;
