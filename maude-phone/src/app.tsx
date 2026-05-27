import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Outlet } from "react-router-dom";
import "./index.css";
import { Home } from "./pages/Home/Home";
import { Chat } from "./pages/Maude/Chat";
import { Voice } from "./pages/Maude/Voice";
import { Terminal } from "./pages/Terminal/Terminal";
import { Browser } from "./pages/Browser/Browser";
import { Messages } from "./pages/Messages/Messages";
import { Files } from "./pages/Files/Files";
import { Settings } from "./pages/Settings/Settings";
import { Collab } from "./pages/Collab/Collab";
import { CommandCenter } from "./pages/CommandCenter/CommandCenter";
import { startPresenceHeartbeat } from "./pages/Collab/useCollab";
import { TabBar } from "./components/TabBar/TabBar";

// Start presence heartbeat at app load (auto-detects device type)
startPresenceHeartbeat();

// MAUDE is a live gateway app. Remove any legacy PWA service worker so
// Safari cannot serve stale app shells or intercept chat requests.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.getRegistrations()
    .then((regs) => Promise.all(regs.map((reg) => reg.unregister())))
    .catch(() => {});
}

function AppLayout() {
  return (
    <div className="flex h-[100dvh] flex-col bg-maude-bg safe-top">
      <div className="min-h-0 flex-1 overflow-hidden">
        <Outlet />
      </div>
      <TabBar />
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { path: "/", element: <Home /> },
      { path: "/maude", element: <Chat /> },
      { path: "/maude/voice", element: <Voice /> },
      { path: "/terminal", element: <Terminal /> },
      { path: "/browser", element: <Browser /> },
      { path: "/messages", element: <Messages /> },
      { path: "/files", element: <Files /> },
      { path: "/collab", element: <Collab /> },
      { path: "/command-center", element: <CommandCenter /> },
      { path: "/settings", element: <Settings /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <RouterProvider router={router} />,
);
