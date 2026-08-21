import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// No <React.StrictMode>: in dev it double-fires every effect, which doubles
// the request volume against the backend's per-minute rate limit.
ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
