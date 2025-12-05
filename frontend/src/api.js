// frontend/src/api.js
export async function uploadImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Server error");
  }

  return res.json(); // { label, bbox: [x1,y1,x2,y2], image_width, image_height }
}
