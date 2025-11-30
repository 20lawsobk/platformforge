import { useState, useCallback, useMemo, useEffect } from "react";
import { useParams, Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
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
  Loader2,
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
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import type { ProjectFile, ConsoleLog } from "@shared/schema";

interface FileNode {
  id?: string;
  name: string;
  path: string;
  type: "file" | "folder";
  content?: string;
  children?: FileNode[];
}

interface ConsoleMessage {
  id: string;
  type: "log" | "error" | "warn" | "info" | "system" | "success";
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

const getConsoleIcon = (type: ConsoleMessage["type"]) => {
  const iconClass = "h-3 w-3 shrink-0 mt-0.5";
  switch (type) {
    case "error":
      return <AlertCircle className={cn(iconClass, "text-red-400")} />;
    case "warn":
      return <AlertCircle className={cn(iconClass, "text-yellow-400")} />;
    case "info":
      return <Info className={cn(iconClass, "text-blue-400")} />;
    case "system":
      return <Terminal className={cn(iconClass, "text-green-400")} />;
    case "success":
      return <CheckCircle2 className={cn(iconClass, "text-green-400")} />;
    default:
      return <ChevronRight className={cn(iconClass, "text-zinc-500")} />;
  }
};

function buildFileTree(files: ProjectFile[]): FileNode[] {
  const nodeMap = new Map<string, FileNode>();
  const rootNodes: FileNode[] = [];

  const sortedFiles = [...files].sort((a, b) => {
    const aDepth = (a.path.match(/\//g) || []).length;
    const bDepth = (b.path.match(/\//g) || []).length;
    return aDepth - bDepth;
  });

  for (const file of sortedFiles) {
    const node: FileNode = {
      id: file.id,
      name: file.name,
      path: file.path,
      type: file.isFolder ? "folder" : "file",
      content: file.content || "",
      children: file.isFolder ? [] : undefined,
    };

    nodeMap.set(file.path, node);

    if (!file.parentPath || file.parentPath === "/" || file.parentPath === "") {
      rootNodes.push(node);
    } else {
      const parent = nodeMap.get(file.parentPath);
      if (parent && parent.children) {
        parent.children.push(node);
      } else {
        rootNodes.push(node);
      }
    }
  }

  const sortNodes = (nodes: FileNode[]): FileNode[] => {
    return nodes.sort((a, b) => {
      if (a.type === b.type) return a.name.localeCompare(b.name);
      return a.type === "folder" ? -1 : 1;
    }).map(node => {
      if (node.children) {
        node.children = sortNodes(node.children);
      }
      return node;
    });
  };

  return sortNodes(rootNodes);
}

function findFileInTree(nodes: FileNode[], path: string): FileNode | null {
  for (const node of nodes) {
    if (node.path === path) return node;
    if (node.children) {
      const found = findFileInTree(node.children, path);
      if (found) return found;
    }
  }
  return null;
}

export default function IDE() {
  const params = useParams<{ projectId: string }>();
  const projectId = params.projectId || "demo";
  const { toast } = useToast();

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [openFiles, setOpenFiles] = useState<string[]>([]);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [fileContents, setFileContents] = useState<Record<string, string>>({});
  const [unsavedChanges, setUnsavedChanges] = useState<Set<string>>(new Set());
  const [rightPanelTab, setRightPanelTab] = useState<"preview" | "console">("console");
  const [localConsoleMessages, setLocalConsoleMessages] = useState<ConsoleMessage[]>([
    {
      id: "init",
      type: "system",
      message: "IDE initialized. Ready to run your project.",
      timestamp: new Date(),
    },
  ]);
  const [editorFontSize, setEditorFontSize] = useState<string>("14");
  const [editorTheme, setEditorTheme] = useState<string>("vs-dark");

  const [dialogState, setDialogState] = useState<{
    open: boolean;
    type: "file" | "folder" | "rename";
    parentPath: string;
    currentName?: string;
    currentPath?: string;
    fileId?: string;
  }>({
    open: false,
    type: "file",
    parentPath: "/",
  });
  const [newItemName, setNewItemName] = useState("");

  const { data: filesData, isLoading: isLoadingFiles } = useQuery<ProjectFile[]>({
    queryKey: ["/api/projects", projectId, "files"],
    enabled: projectId !== "demo",
  });

  const { data: consoleLogsData } = useQuery<ConsoleLog[]>({
    queryKey: ["/api/projects", projectId, "console"],
    enabled: projectId !== "demo",
  });

  const { data: project } = useQuery({
    queryKey: ["/api/projects", projectId],
    enabled: projectId !== "demo",
  });

  const projectName = (project as { name?: string })?.name || `Project ${projectId}`;

  const fileTree = useMemo(() => {
    if (filesData && filesData.length > 0) {
      return buildFileTree(filesData);
    }
    return [];
  }, [filesData]);

  const consoleMessages = useMemo(() => {
    if (consoleLogsData && consoleLogsData.length > 0) {
      return consoleLogsData.map(log => ({
        id: log.id,
        type: log.type as ConsoleMessage["type"],
        message: log.message,
        timestamp: new Date(log.timestamp),
      }));
    }
    return localConsoleMessages;
  }, [consoleLogsData, localConsoleMessages]);

  useEffect(() => {
    if (fileTree.length > 0 && openFiles.length === 0) {
      const folders = fileTree.filter(n => n.type === "folder");
      if (folders.length > 0) {
        setExpandedFolders(new Set(folders.slice(0, 2).map(f => f.path)));
      }
    }
  }, [fileTree, openFiles.length]);

  const saveFileMutation = useMutation({
    mutationFn: async ({ fileId, content, path }: { fileId: string; content: string; path: string }) => {
      const response = await apiRequest("PUT", `/api/projects/${projectId}/files/${fileId}`, { content });
      return response.json();
    },
    onSuccess: (_, variables) => {
      setUnsavedChanges(prev => {
        const next = new Set(prev);
        next.delete(variables.path);
        return next;
      });
      toast({
        title: "File saved",
        description: `Successfully saved ${variables.path.split("/").pop()}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId, "files"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Save failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const createFileMutation = useMutation({
    mutationFn: async (data: { path: string; name: string; content: string; isFolder: boolean; parentPath: string }) => {
      const response = await apiRequest("POST", `/api/projects/${projectId}/files`, data);
      return response.json();
    },
    onSuccess: (file, variables) => {
      toast({
        title: variables.isFolder ? "Folder created" : "File created",
        description: `Created ${variables.name}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId, "files"] });
      if (!variables.isFolder) {
        openFile(variables.path);
      }
      if (variables.parentPath && variables.parentPath !== "/") {
        setExpandedFolders(prev => new Set(prev).add(variables.parentPath));
      }
    },
    onError: (error: Error) => {
      toast({
        title: "Creation failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const deleteFileMutation = useMutation({
    mutationFn: async ({ fileId }: { fileId: string }) => {
      const response = await apiRequest("DELETE", `/api/projects/${projectId}/files/${fileId}`);
      return response.json();
    },
    onSuccess: (_, variables) => {
      toast({
        title: "Deleted",
        description: "Item has been deleted",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId, "files"] });
    },
    onError: (error: Error) => {
      toast({
        title: "Delete failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const runProjectMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", `/api/projects/${projectId}/run`);
      return response.json();
    },
    onSuccess: (data) => {
      if (data.logs) {
        setLocalConsoleMessages(prev => [
          ...prev,
          ...data.logs.map((log: { type: string; message: string; timestamp: string }) => ({
            id: Date.now().toString() + Math.random(),
            type: log.type as ConsoleMessage["type"],
            message: log.message,
            timestamp: new Date(log.timestamp),
          })),
        ]);
      }
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId, "console"] });
      toast({
        title: "Project running",
        description: "Build completed successfully",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Run failed",
        description: error.message,
        variant: "destructive",
      });
      setLocalConsoleMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          type: "error",
          message: `Failed to run: ${error.message}`,
          timestamp: new Date(),
        },
      ]);
    },
  });

  const clearConsoleMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("DELETE", `/api/projects/${projectId}/console`);
      return response.json();
    },
    onSuccess: () => {
      setLocalConsoleMessages([]);
      queryClient.invalidateQueries({ queryKey: ["/api/projects", projectId, "console"] });
    },
  });

  const findFileContent = useCallback(
    (path: string): string => {
      if (fileContents[path] !== undefined) {
        return fileContents[path];
      }
      const file = findFileInTree(fileTree, path);
      return file?.content || "";
    },
    [fileTree, fileContents]
  );

  const findFileId = useCallback(
    (path: string): string | undefined => {
      const file = findFileInTree(fileTree, path);
      return file?.id;
    },
    [fileTree]
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
    setFileContents((prev) => {
      const next = { ...prev };
      delete next[path];
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
    const fileId = findFileId(path);
    if (fileId) {
      const content = fileContents[path] ?? findFileContent(path);
      saveFileMutation.mutate({ fileId, content, path });
    } else {
      toast({
        title: "Save failed",
        description: "Could not find file to save",
        variant: "destructive",
      });
    }
  };

  const saveAllFiles = () => {
    unsavedChanges.forEach((path) => {
      saveFile(path);
    });
  };

  const addLocalConsoleMessage = (type: ConsoleMessage["type"], message: string) => {
    setLocalConsoleMessages((prev) => [
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
    setRightPanelTab("console");
    addLocalConsoleMessage("system", "Starting development server...");
    runProjectMutation.mutate();
  };

  const handleCreateItem = () => {
    if (!newItemName.trim()) return;

    const newPath =
      dialogState.parentPath === "/"
        ? `/${newItemName}`
        : `${dialogState.parentPath}/${newItemName}`;

    createFileMutation.mutate({
      path: newPath,
      name: newItemName,
      content: dialogState.type === "folder" ? "" : "",
      isFolder: dialogState.type === "folder",
      parentPath: dialogState.parentPath === "/" ? "" : dialogState.parentPath,
    });

    setDialogState({ open: false, type: "file", parentPath: "/" });
    setNewItemName("");
  };

  const handleRename = () => {
    if (!newItemName.trim() || !dialogState.currentPath || !dialogState.fileId) return;
    
    const parentPath = dialogState.currentPath.substring(
      0,
      dialogState.currentPath.lastIndexOf("/")
    );
    const newPath = parentPath ? `${parentPath}/${newItemName}` : `/${newItemName}`;

    saveFileMutation.mutate({
      fileId: dialogState.fileId,
      content: findFileContent(dialogState.currentPath),
      path: newPath,
    });

    if (openFiles.includes(dialogState.currentPath)) {
      setOpenFiles((prev) =>
        prev.map((f) => (f === dialogState.currentPath ? newPath : f))
      );
    }
    if (activeFile === dialogState.currentPath) {
      setActiveFile(newPath);
    }

    setDialogState({ open: false, type: "file", parentPath: "/" });
    setNewItemName("");
  };

  const handleDelete = (path: string, fileId?: string) => {
    if (fileId) {
      deleteFileMutation.mutate({ fileId });
      if (openFiles.includes(path)) {
        setOpenFiles((prev) => prev.filter((f) => f !== path));
      }
      if (activeFile === path) {
        const remainingFiles = openFiles.filter((f) => f !== path);
        setActiveFile(
          remainingFiles.length > 0 ? remainingFiles[remainingFiles.length - 1] : null
        );
      }
    }
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
                      <FolderOpen className="h-4 w-4 shrink-0 text-yellow-400" />
                    ) : (
                      <Folder className="h-4 w-4 shrink-0 text-yellow-400" />
                    )}
                  </>
                ) : (
                  <>
                    <span className="w-4" />
                    {getFileIcon(node.name)}
                  </>
                )}
                <span className="truncate">{node.name}</span>
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
                    data-testid={`context-new-file-${node.path}`}
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
                    data-testid={`context-new-folder-${node.path}`}
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
                    fileId: node.id,
                  });
                  setNewItemName(node.name);
                }}
                data-testid={`context-rename-${node.path}`}
              >
                <Edit3 className="h-4 w-4 mr-2" />
                Rename
              </ContextMenuItem>
              <ContextMenuSeparator />
              <ContextMenuItem
                className="text-destructive focus:text-destructive"
                onClick={() => handleDelete(node.path, node.id)}
                data-testid={`context-delete-${node.path}`}
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

  const isDemoMode = projectId === "demo";
  const isLoading = !isDemoMode && isLoadingFiles;
  const hasNoFiles = !isDemoMode && !isLoadingFiles && fileTree.length === 0;

  return (
    <div className="flex flex-col h-screen bg-background" data-testid="ide-container">
      <header className="flex items-center justify-between gap-4 px-4 py-2 border-b bg-card/50">
        <div className="flex items-center gap-3">
          <Link href="/dashboard">
            <Button variant="ghost" size="icon" data-testid="button-back">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div className="flex items-center gap-2">
            <span className="font-semibold" data-testid="text-project-name">{projectName}</span>
            {unsavedChanges.size > 0 && (
              <span className="text-xs text-muted-foreground" data-testid="text-unsaved-count">
                ({unsavedChanges.size} unsaved)
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => activeFile && saveFile(activeFile)}
            disabled={!activeFile || !unsavedChanges.has(activeFile || "") || saveFileMutation.isPending}
            data-testid="button-save"
          >
            {saveFileMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-2" />
            )}
            Save
          </Button>
          <Button
            size="sm"
            onClick={runProject}
            disabled={runProjectMutation.isPending}
            data-testid="button-run"
          >
            {runProjectMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Run
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" data-testid="button-settings">
                <Settings className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
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
                {isLoading ? (
                  <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : isDemoMode ? (
                  <div className="flex flex-col items-center justify-center p-8 text-center">
                    <File className="h-8 w-8 text-muted-foreground mb-2" />
                    <p className="text-sm text-muted-foreground">Demo mode</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Create a project to start editing
                    </p>
                  </div>
                ) : hasNoFiles ? (
                  <div className="flex flex-col items-center justify-center p-8 text-center">
                    <FolderPlus className="h-8 w-8 text-muted-foreground mb-2" />
                    <p className="text-sm text-muted-foreground">No files yet</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Click the + button to create a file
                    </p>
                  </div>
                ) : (
                  renderFileTree(fileTree)
                )}
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
                          msg.type === "success" && "text-green-400",
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
                    onClick={() => {
                      if (projectId !== "demo") {
                        clearConsoleMutation.mutate();
                      } else {
                        setLocalConsoleMessages([]);
                      }
                    }}
                    disabled={clearConsoleMutation.isPending}
                    data-testid="button-clear-console"
                  >
                    {clearConsoleMutation.isPending && (
                      <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    )}
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
              disabled={createFileMutation.isPending || saveFileMutation.isPending}
              data-testid="button-confirm"
            >
              {(createFileMutation.isPending || saveFileMutation.isPending) && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              {dialogState.type === "rename" ? "Rename" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
