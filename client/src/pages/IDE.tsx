import { useState, useCallback } from "react";
import { useParams, Link } from "wouter";
import Editor from "@monaco-editor/react";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import {
  Play,
  Save,
  Settings,
  ChevronRight,
  ChevronDown,
  X,
  File,
  Folder,
  FolderOpen,
  Plus,
  FilePlus,
  FolderPlus,
  Trash2,
  Edit3,
  ArrowLeft,
  Terminal,
  Eye,
  AlertCircle,
  CheckCircle2,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
} from "@/components/ui/dropdown-menu";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

interface FileNode {
  name: string;
  path: string;
  type: "file" | "folder";
  content?: string;
  children?: FileNode[];
}

interface ConsoleMessage {
  id: string;
  type: "log" | "error" | "warn" | "info" | "system";
  message: string;
  timestamp: Date;
}

const getFileIcon = (filename: string) => {
  const ext = filename.split(".").pop()?.toLowerCase();
  const iconClass = "h-4 w-4 shrink-0";
  
  switch (ext) {
    case "js":
    case "jsx":
      return <File className={cn(iconClass, "text-yellow-400")} />;
    case "ts":
    case "tsx":
      return <File className={cn(iconClass, "text-blue-400")} />;
    case "css":
    case "scss":
    case "sass":
      return <File className={cn(iconClass, "text-pink-400")} />;
    case "html":
      return <File className={cn(iconClass, "text-orange-400")} />;
    case "json":
      return <File className={cn(iconClass, "text-yellow-300")} />;
    case "md":
      return <File className={cn(iconClass, "text-gray-400")} />;
    case "py":
      return <File className={cn(iconClass, "text-green-400")} />;
    case "go":
      return <File className={cn(iconClass, "text-cyan-400")} />;
    case "rs":
      return <File className={cn(iconClass, "text-orange-500")} />;
    default:
      return <File className={cn(iconClass, "text-muted-foreground")} />;
  }
};

const getLanguageFromExtension = (filename: string): string => {
  const ext = filename.split(".").pop()?.toLowerCase();
  
  switch (ext) {
    case "js":
      return "javascript";
    case "jsx":
      return "javascript";
    case "ts":
      return "typescript";
    case "tsx":
      return "typescript";
    case "css":
      return "css";
    case "scss":
      return "scss";
    case "html":
      return "html";
    case "json":
      return "json";
    case "md":
      return "markdown";
    case "py":
      return "python";
    case "go":
      return "go";
    case "rs":
      return "rust";
    case "yaml":
    case "yml":
      return "yaml";
    case "sh":
    case "bash":
      return "shell";
    default:
      return "plaintext";
  }
};

const initialFileSystem: FileNode[] = [
  {
    name: "src",
    path: "/src",
    type: "folder",
    children: [
      {
        name: "components",
        path: "/src/components",
        type: "folder",
        children: [
          {
            name: "Button.tsx",
            path: "/src/components/Button.tsx",
            type: "file",
            content: `import { cn } from "../lib/utils";

interface ButtonProps {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  onClick?: () => void;
  className?: string;
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  onClick,
  className,
}: ButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "rounded-md font-medium transition-colors",
        variant === "primary" && "bg-blue-500 text-white hover:bg-blue-600",
        variant === "secondary" && "bg-gray-200 text-gray-900 hover:bg-gray-300",
        variant === "ghost" && "hover:bg-gray-100",
        size === "sm" && "px-2 py-1 text-sm",
        size === "md" && "px-4 py-2",
        size === "lg" && "px-6 py-3 text-lg",
        className
      )}
    >
      {children}
    </button>
  );
}`,
          },
          {
            name: "Card.tsx",
            path: "/src/components/Card.tsx",
            type: "file",
            content: `interface CardProps {
  title: string;
  children: React.ReactNode;
}

export function Card({ title, children }: CardProps) {
  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <h3 className="mb-2 text-lg font-semibold">{title}</h3>
      {children}
    </div>
  );
}`,
          },
        ],
      },
      {
        name: "lib",
        path: "/src/lib",
        type: "folder",
        children: [
          {
            name: "utils.ts",
            path: "/src/lib/utils.ts",
            type: "file",
            content: `import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date);
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}`,
          },
        ],
      },
      {
        name: "App.tsx",
        path: "/src/App.tsx",
        type: "file",
        content: `import { Button } from "./components/Button";
import { Card } from "./components/Card";

function App() {
  return (
    <div className="min-h-screen bg-background p-8">
      <h1 className="mb-8 text-3xl font-bold">My Application</h1>
      
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card title="Welcome">
          <p className="text-muted-foreground">
            This is your new application. Start building something amazing!
          </p>
          <Button className="mt-4">Get Started</Button>
        </Card>
        
        <Card title="Features">
          <ul className="list-inside list-disc text-muted-foreground">
            <li>Modern React with TypeScript</li>
            <li>Tailwind CSS for styling</li>
            <li>Component-based architecture</li>
          </ul>
        </Card>
        
        <Card title="Documentation">
          <p className="text-muted-foreground">
            Check out the docs to learn more about the available components.
          </p>
          <Button variant="secondary" className="mt-4">
            View Docs
          </Button>
        </Card>
      </div>
    </div>
  );
}

export default App;`,
      },
      {
        name: "main.tsx",
        path: "/src/main.tsx",
        type: "file",
        content: `import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);`,
      },
      {
        name: "index.css",
        path: "/src/index.css",
        type: "file",
        content: `@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --card: 0 0% 100%;
  --card-foreground: 222.2 84% 4.9%;
  --primary: 221.2 83.2% 53.3%;
  --primary-foreground: 210 40% 98%;
  --muted: 210 40% 96.1%;
  --muted-foreground: 215.4 16.3% 46.9%;
}

.dark {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --card: 222.2 84% 4.9%;
  --card-foreground: 210 40% 98%;
  --primary: 217.2 91.2% 59.8%;
  --primary-foreground: 222.2 47.4% 11.2%;
  --muted: 217.2 32.6% 17.5%;
  --muted-foreground: 215 20.2% 65.1%;
}

body {
  font-family: system-ui, -apple-system, sans-serif;
}`,
      },
    ],
  },
  {
    name: "public",
    path: "/public",
    type: "folder",
    children: [
      {
        name: "index.html",
        path: "/public/index.html",
        type: "file",
        content: `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>My Application</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>`,
      },
      {
        name: "favicon.ico",
        path: "/public/favicon.ico",
        type: "file",
        content: "// Binary file",
      },
    ],
  },
  {
    name: "package.json",
    path: "/package.json",
    type: "file",
    content: `{
  "name": "my-application",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}`,
  },
  {
    name: "tsconfig.json",
    path: "/tsconfig.json",
    type: "file",
    content: `{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}`,
  },
  {
    name: "README.md",
    path: "/README.md",
    type: "file",
    content: `# My Application

A modern web application built with React and TypeScript.

## Getting Started

\`\`\`bash
npm install
npm run dev
\`\`\`

## Features

- React 18 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Component-based architecture

## Project Structure

\`\`\`
src/
├── components/    # Reusable UI components
├── lib/          # Utility functions
├── App.tsx       # Main application component
├── main.tsx      # Application entry point
└── index.css     # Global styles
\`\`\`

## License

MIT
`,
  },
];

export default function IDE() {
  const params = useParams<{ projectId: string }>();
  const projectId = params.projectId || "demo";

  const [fileSystem, setFileSystem] = useState<FileNode[]>(initialFileSystem);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(
    new Set(["/src", "/src/components"])
  );
  const [openFiles, setOpenFiles] = useState<string[]>(["/src/App.tsx"]);
  const [activeFile, setActiveFile] = useState<string | null>("/src/App.tsx");
  const [fileContents, setFileContents] = useState<Record<string, string>>({});
  const [unsavedChanges, setUnsavedChanges] = useState<Set<string>>(new Set());
  const [rightPanelTab, setRightPanelTab] = useState<"preview" | "console">("console");
  const [consoleMessages, setConsoleMessages] = useState<ConsoleMessage[]>([
    {
      id: "1",
      type: "system",
      message: "IDE initialized. Ready to run your project.",
      timestamp: new Date(),
    },
  ]);
  const [editorFontSize, setEditorFontSize] = useState<string>("14");
  const [editorTheme, setEditorTheme] = useState<string>("vs-dark");
  const [isRunning, setIsRunning] = useState(false);

  const [dialogState, setDialogState] = useState<{
    open: boolean;
    type: "file" | "folder" | "rename";
    parentPath: string;
    currentName?: string;
    currentPath?: string;
  }>({
    open: false,
    type: "file",
    parentPath: "/",
  });
  const [newItemName, setNewItemName] = useState("");

  const projectName = `Project ${projectId}`;

  const findFileContent = useCallback(
    (path: string): string => {
      if (fileContents[path] !== undefined) {
        return fileContents[path];
      }

      const findInNodes = (nodes: FileNode[]): string | null => {
        for (const node of nodes) {
          if (node.path === path && node.type === "file") {
            return node.content || "";
          }
          if (node.children) {
            const found = findInNodes(node.children);
            if (found !== null) return found;
          }
        }
        return null;
      };

      return findInNodes(fileSystem) || "";
    },
    [fileSystem, fileContents]
  );

  const toggleFolder = (path: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const openFile = (path: string) => {
    if (!openFiles.includes(path)) {
      setOpenFiles((prev) => [...prev, path]);
    }
    setActiveFile(path);
  };

  const closeFile = (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenFiles((prev) => prev.filter((f) => f !== path));
    if (activeFile === path) {
      const remainingFiles = openFiles.filter((f) => f !== path);
      setActiveFile(remainingFiles.length > 0 ? remainingFiles[remainingFiles.length - 1] : null);
    }
    setUnsavedChanges((prev) => {
      const next = new Set(prev);
      next.delete(path);
      return next;
    });
  };

  const handleEditorChange = (value: string | undefined, path: string) => {
    if (value !== undefined) {
      setFileContents((prev) => ({ ...prev, [path]: value }));
      setUnsavedChanges((prev) => new Set(prev).add(path));
    }
  };

  const saveFile = (path: string) => {
    setUnsavedChanges((prev) => {
      const next = new Set(prev);
      next.delete(path);
      return next;
    });
    addConsoleMessage("info", `Saved: ${path}`);
  };

  const saveAllFiles = () => {
    unsavedChanges.forEach((path) => {
      saveFile(path);
    });
    addConsoleMessage("system", `Saved ${unsavedChanges.size} file(s)`);
  };

  const addConsoleMessage = (type: ConsoleMessage["type"], message: string) => {
    setConsoleMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        type,
        message,
        timestamp: new Date(),
      },
    ]);
  };

  const runProject = () => {
    setIsRunning(true);
    setRightPanelTab("console");
    addConsoleMessage("system", "Starting development server...");

    setTimeout(() => {
      addConsoleMessage("log", "Compiling...");
    }, 500);

    setTimeout(() => {
      addConsoleMessage("log", "Bundling modules...");
    }, 1000);

    setTimeout(() => {
      addConsoleMessage("info", "Build completed in 1.2s");
    }, 1500);

    setTimeout(() => {
      addConsoleMessage("log", "Server running at http://localhost:5173");
      setIsRunning(false);
    }, 2000);
  };

  const addNodeToFileSystem = (
    nodes: FileNode[],
    parentPath: string,
    newNode: FileNode
  ): FileNode[] => {
    return nodes.map((node) => {
      if (node.path === parentPath && node.type === "folder") {
        return {
          ...node,
          children: [...(node.children || []), newNode].sort((a, b) => {
            if (a.type === b.type) return a.name.localeCompare(b.name);
            return a.type === "folder" ? -1 : 1;
          }),
        };
      }
      if (node.children) {
        return {
          ...node,
          children: addNodeToFileSystem(node.children, parentPath, newNode),
        };
      }
      return node;
    });
  };

  const deleteNodeFromFileSystem = (nodes: FileNode[], path: string): FileNode[] => {
    return nodes
      .filter((node) => node.path !== path)
      .map((node) => {
        if (node.children) {
          return {
            ...node,
            children: deleteNodeFromFileSystem(node.children, path),
          };
        }
        return node;
      });
  };

  const renameNodeInFileSystem = (
    nodes: FileNode[],
    oldPath: string,
    newName: string
  ): FileNode[] => {
    return nodes.map((node) => {
      if (node.path === oldPath) {
        const parentPath = oldPath.substring(0, oldPath.lastIndexOf("/"));
        const newPath = parentPath ? `${parentPath}/${newName}` : `/${newName}`;
        return {
          ...node,
          name: newName,
          path: newPath,
        };
      }
      if (node.children) {
        return {
          ...node,
          children: renameNodeInFileSystem(node.children, oldPath, newName),
        };
      }
      return node;
    });
  };

  const handleCreateItem = () => {
    if (!newItemName.trim()) return;

    const newPath =
      dialogState.parentPath === "/"
        ? `/${newItemName}`
        : `${dialogState.parentPath}/${newItemName}`;

    const newNode: FileNode =
      dialogState.type === "folder"
        ? { name: newItemName, path: newPath, type: "folder", children: [] }
        : { name: newItemName, path: newPath, type: "file", content: "" };

    if (dialogState.parentPath === "/") {
      setFileSystem((prev) =>
        [...prev, newNode].sort((a, b) => {
          if (a.type === b.type) return a.name.localeCompare(b.name);
          return a.type === "folder" ? -1 : 1;
        })
      );
    } else {
      setFileSystem((prev) => addNodeToFileSystem(prev, dialogState.parentPath, newNode));
      setExpandedFolders((prev) => new Set(prev).add(dialogState.parentPath));
    }

    if (dialogState.type === "file") {
      openFile(newPath);
    }

    addConsoleMessage(
      "info",
      `Created ${dialogState.type}: ${newPath}`
    );
    setDialogState({ open: false, type: "file", parentPath: "/" });
    setNewItemName("");
  };

  const handleRename = () => {
    if (!newItemName.trim() || !dialogState.currentPath) return;

    setFileSystem((prev) =>
      renameNodeInFileSystem(prev, dialogState.currentPath!, newItemName)
    );

    const parentPath = dialogState.currentPath.substring(
      0,
      dialogState.currentPath.lastIndexOf("/")
    );
    const newPath = parentPath ? `${parentPath}/${newItemName}` : `/${newItemName}`;

    if (openFiles.includes(dialogState.currentPath)) {
      setOpenFiles((prev) =>
        prev.map((f) => (f === dialogState.currentPath ? newPath : f))
      );
    }
    if (activeFile === dialogState.currentPath) {
      setActiveFile(newPath);
    }

    addConsoleMessage("info", `Renamed to: ${newPath}`);
    setDialogState({ open: false, type: "file", parentPath: "/" });
    setNewItemName("");
  };

  const handleDelete = (path: string, name: string) => {
    setFileSystem((prev) => deleteNodeFromFileSystem(prev, path));
    if (openFiles.includes(path)) {
      setOpenFiles((prev) => prev.filter((f) => f !== path));
    }
    if (activeFile === path) {
      const remainingFiles = openFiles.filter((f) => f !== path);
      setActiveFile(
        remainingFiles.length > 0 ? remainingFiles[remainingFiles.length - 1] : null
      );
    }
    addConsoleMessage("warn", `Deleted: ${path}`);
  };

  const renderFileTree = (nodes: FileNode[], depth: number = 0) => {
    return nodes.map((node) => {
      const isExpanded = expandedFolders.has(node.path);
      const isActive = activeFile === node.path;

      return (
        <div key={node.path}>
          <ContextMenu>
            <ContextMenuTrigger asChild>
              <div
                className={cn(
                  "flex items-center gap-1 py-1 px-2 cursor-pointer text-sm hover-elevate rounded-sm",
                  isActive && "bg-accent"
                )}
                style={{ paddingLeft: `${depth * 12 + 8}px` }}
                onClick={() => {
                  if (node.type === "folder") {
                    toggleFolder(node.path);
                  } else {
                    openFile(node.path);
                  }
                }}
                data-testid={`tree-item-${node.path.replace(/\//g, "-")}`}
              >
                {node.type === "folder" ? (
                  <>
                    {isExpanded ? (
                      <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                    )}
                    {isExpanded ? (
                      <FolderOpen className="h-4 w-4 shrink-0 text-yellow-500" />
                    ) : (
                      <Folder className="h-4 w-4 shrink-0 text-yellow-500" />
                    )}
                  </>
                ) : (
                  <>
                    <span className="w-4" />
                    {getFileIcon(node.name)}
                  </>
                )}
                <span className="truncate">{node.name}</span>
                {unsavedChanges.has(node.path) && (
                  <span className="ml-auto h-2 w-2 rounded-full bg-blue-400" />
                )}
              </div>
            </ContextMenuTrigger>
            <ContextMenuContent>
              {node.type === "folder" && (
                <>
                  <ContextMenuItem
                    onClick={() => {
                      setDialogState({
                        open: true,
                        type: "file",
                        parentPath: node.path,
                      });
                      setNewItemName("");
                    }}
                    data-testid={`context-new-file-${node.path.replace(/\//g, "-")}`}
                  >
                    <FilePlus className="h-4 w-4 mr-2" />
                    New File
                  </ContextMenuItem>
                  <ContextMenuItem
                    onClick={() => {
                      setDialogState({
                        open: true,
                        type: "folder",
                        parentPath: node.path,
                      });
                      setNewItemName("");
                    }}
                    data-testid={`context-new-folder-${node.path.replace(/\//g, "-")}`}
                  >
                    <FolderPlus className="h-4 w-4 mr-2" />
                    New Folder
                  </ContextMenuItem>
                  <ContextMenuSeparator />
                </>
              )}
              <ContextMenuItem
                onClick={() => {
                  setDialogState({
                    open: true,
                    type: "rename",
                    parentPath: node.path.substring(0, node.path.lastIndexOf("/")),
                    currentName: node.name,
                    currentPath: node.path,
                  });
                  setNewItemName(node.name);
                }}
                data-testid={`context-rename-${node.path.replace(/\//g, "-")}`}
              >
                <Edit3 className="h-4 w-4 mr-2" />
                Rename
              </ContextMenuItem>
              <ContextMenuItem
                className="text-destructive focus:text-destructive"
                onClick={() => handleDelete(node.path, node.name)}
                data-testid={`context-delete-${node.path.replace(/\//g, "-")}`}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
          {node.type === "folder" && isExpanded && node.children && (
            <div>{renderFileTree(node.children, depth + 1)}</div>
          )}
        </div>
      );
    });
  };

  const getConsoleIcon = (type: ConsoleMessage["type"]) => {
    switch (type) {
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />;
      case "warn":
        return <AlertCircle className="h-4 w-4 text-yellow-400 shrink-0" />;
      case "info":
        return <Info className="h-4 w-4 text-blue-400 shrink-0" />;
      case "system":
        return <CheckCircle2 className="h-4 w-4 text-green-400 shrink-0" />;
      default:
        return <Terminal className="h-4 w-4 text-muted-foreground shrink-0" />;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background" data-testid="ide-container">
      <header className="flex items-center justify-between gap-4 px-4 h-12 border-b bg-card shrink-0">
        <div className="flex items-center gap-4">
          <Link href="/dashboard">
            <Button variant="ghost" size="sm" data-testid="button-back-dashboard">
              <ArrowLeft className="h-4 w-4 mr-1" />
              Dashboard
            </Button>
          </Link>
          <div className="h-4 w-px bg-border" />
          <span className="font-semibold text-sm" data-testid="text-project-name">
            {projectName}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="default"
            size="sm"
            onClick={runProject}
            disabled={isRunning}
            data-testid="button-run"
          >
            <Play className="h-4 w-4 mr-1" />
            {isRunning ? "Running..." : "Run"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={saveAllFiles}
            disabled={unsavedChanges.size === 0}
            data-testid="button-save-all"
          >
            <Save className="h-4 w-4 mr-1" />
            Save All
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" data-testid="button-settings">
                <Settings className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>Editor Settings</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
                Font Size
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={editorFontSize}
                onValueChange={setEditorFontSize}
              >
                <DropdownMenuRadioItem value="12" data-testid="option-font-12">
                  12px
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="14" data-testid="option-font-14">
                  14px
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="16" data-testid="option-font-16">
                  16px
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="18" data-testid="option-font-18">
                  18px
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
                Theme
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup
                value={editorTheme}
                onValueChange={setEditorTheme}
              >
                <DropdownMenuRadioItem value="vs-dark" data-testid="option-theme-dark">
                  Dark
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="light" data-testid="option-theme-light">
                  Light
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="hc-black" data-testid="option-theme-hc">
                  High Contrast
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <ResizablePanelGroup direction="horizontal" className="flex-1">
        <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
          <div className="flex flex-col h-full bg-card/50">
            <div className="flex items-center justify-between gap-2 p-2 border-b">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Files
              </span>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={() => {
                    setDialogState({ open: true, type: "file", parentPath: "/" });
                    setNewItemName("");
                  }}
                  data-testid="button-create-file"
                >
                  <FilePlus className="h-3.5 w-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  onClick={() => {
                    setDialogState({ open: true, type: "folder", parentPath: "/" });
                    setNewItemName("");
                  }}
                  data-testid="button-create-folder"
                >
                  <FolderPlus className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
            <ScrollArea className="flex-1">
              <div className="py-1" data-testid="file-tree">
                {renderFileTree(fileSystem)}
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize={50} minSize={30}>
          <div className="flex flex-col h-full">
            <div className="flex items-center border-b bg-card/50 overflow-x-auto">
              {openFiles.map((filePath) => {
                const fileName = filePath.split("/").pop() || filePath;
                const isActive = activeFile === filePath;
                const hasChanges = unsavedChanges.has(filePath);

                return (
                  <div
                    key={filePath}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 border-r cursor-pointer text-sm hover-elevate",
                      isActive ? "bg-background" : "bg-card/50"
                    )}
                    onClick={() => setActiveFile(filePath)}
                    data-testid={`tab-${filePath.replace(/\//g, "-")}`}
                  >
                    {getFileIcon(fileName)}
                    <span className={cn(hasChanges && "italic")}>
                      {fileName}
                      {hasChanges && " *"}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-4 w-4 ml-1 opacity-60 hover:opacity-100"
                      onClick={(e) => closeFile(filePath, e)}
                      data-testid={`tab-close-${filePath.replace(/\//g, "-")}`}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                );
              })}
            </div>

            <div className="flex-1">
              {activeFile ? (
                <Editor
                  height="100%"
                  language={getLanguageFromExtension(activeFile)}
                  value={findFileContent(activeFile)}
                  theme={editorTheme}
                  onChange={(value) => handleEditorChange(value, activeFile)}
                  options={{
                    fontSize: parseInt(editorFontSize),
                    minimap: { enabled: true },
                    scrollBeyondLastLine: false,
                    wordWrap: "on",
                    automaticLayout: true,
                    tabSize: 2,
                    lineNumbers: "on",
                    renderLineHighlight: "line",
                    cursorBlinking: "smooth",
                    smoothScrolling: true,
                  }}
                  data-testid="monaco-editor"
                />
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <div className="text-center">
                    <File className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Select a file to start editing</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        <ResizablePanel defaultSize={30} minSize={20} maxSize={50}>
          <div className="flex flex-col h-full">
            <Tabs
              value={rightPanelTab}
              onValueChange={(v) => setRightPanelTab(v as "preview" | "console")}
              className="flex flex-col h-full"
            >
              <TabsList className="w-full justify-start rounded-none border-b bg-card/50 p-0 h-auto">
                <TabsTrigger
                  value="preview"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-2"
                  data-testid="tab-preview"
                >
                  <Eye className="h-4 w-4 mr-2" />
                  Preview
                </TabsTrigger>
                <TabsTrigger
                  value="console"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-2"
                  data-testid="tab-console"
                >
                  <Terminal className="h-4 w-4 mr-2" />
                  Console
                </TabsTrigger>
              </TabsList>

              <TabsContent value="preview" className="flex-1 m-0 p-0">
                <div className="flex items-center justify-center h-full bg-muted/30">
                  <div className="text-center p-8">
                    <Eye className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <p className="text-muted-foreground mb-2">
                      Preview not available
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Click "Run" to start the development server
                    </p>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="console" className="flex-1 m-0 p-0 flex flex-col">
                <ScrollArea className="flex-1 bg-zinc-950">
                  <div className="p-3 font-mono text-xs space-y-1" data-testid="console-output">
                    {consoleMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={cn(
                          "flex items-start gap-2 py-0.5",
                          msg.type === "error" && "text-red-400",
                          msg.type === "warn" && "text-yellow-400",
                          msg.type === "info" && "text-blue-400",
                          msg.type === "system" && "text-green-400",
                          msg.type === "log" && "text-zinc-300"
                        )}
                        data-testid={`console-message-${msg.id}`}
                      >
                        {getConsoleIcon(msg.type)}
                        <span className="text-zinc-500 shrink-0">
                          [{msg.timestamp.toLocaleTimeString()}]
                        </span>
                        <span className="break-all">{msg.message}</span>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
                <div className="border-t p-2 bg-zinc-950">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs"
                    onClick={() => setConsoleMessages([])}
                    data-testid="button-clear-console"
                  >
                    Clear Console
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>

      <Dialog
        open={dialogState.open}
        onOpenChange={(open) =>
          setDialogState((prev) => ({ ...prev, open }))
        }
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {dialogState.type === "rename"
                ? "Rename"
                : `Create New ${dialogState.type === "folder" ? "Folder" : "File"}`}
            </DialogTitle>
            <DialogDescription>
              {dialogState.type === "rename"
                ? "Enter a new name for this item."
                : `Enter a name for the new ${dialogState.type}.`}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={newItemName}
                onChange={(e) => setNewItemName(e.target.value)}
                placeholder={
                  dialogState.type === "folder" ? "folder-name" : "filename.tsx"
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    dialogState.type === "rename" ? handleRename() : handleCreateItem();
                  }
                }}
                data-testid="input-item-name"
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDialogState((prev) => ({ ...prev, open: false }))}
              data-testid="button-cancel"
            >
              Cancel
            </Button>
            <Button
              onClick={dialogState.type === "rename" ? handleRename : handleCreateItem}
              data-testid="button-confirm"
            >
              {dialogState.type === "rename" ? "Rename" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
