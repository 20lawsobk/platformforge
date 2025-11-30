import { useState } from 'react';
import { 
  Key, 
  Lock, 
  Eye, 
  EyeOff, 
  Plus, 
  Pencil, 
  Trash2, 
  AlertTriangle,
  ShieldAlert
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import DashboardLayout from '@/components/DashboardLayout';

type Environment = 'development' | 'production' | 'shared';

interface EnvVariable {
  id: string;
  key: string;
  value: string;
  environment: Environment;
}

interface Secret {
  id: string;
  key: string;
  createdAt: string;
}

const initialEnvVariables: EnvVariable[] = [
  { id: '1', key: 'DATABASE_URL', value: 'postgresql://user:pass@localhost:5432/db', environment: 'production' },
  { id: '2', key: 'API_KEY', value: 'sk_live_abc123xyz789', environment: 'development' },
  { id: '3', key: 'NODE_ENV', value: 'production', environment: 'shared' },
];

const initialSecrets: Secret[] = [
  { id: '1', key: 'STRIPE_SECRET_KEY', createdAt: '2024-11-15' },
  { id: '2', key: 'JWT_SECRET', createdAt: '2024-11-20' },
];

export default function Secrets() {
  const [envVariables, setEnvVariables] = useState<EnvVariable[]>(initialEnvVariables);
  const [secrets, setSecrets] = useState<Secret[]>(initialSecrets);
  const [visibleValues, setVisibleValues] = useState<Set<string>>(new Set());
  
  const [envDialogOpen, setEnvDialogOpen] = useState(false);
  const [secretDialogOpen, setSecretDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteSecretDialogOpen, setDeleteSecretDialogOpen] = useState(false);
  
  const [editingEnv, setEditingEnv] = useState<EnvVariable | null>(null);
  const [editingSecret, setEditingSecret] = useState<Secret | null>(null);
  const [deletingEnv, setDeletingEnv] = useState<EnvVariable | null>(null);
  const [deletingSecret, setDeletingSecret] = useState<Secret | null>(null);
  
  const [newEnvKey, setNewEnvKey] = useState('');
  const [newEnvValue, setNewEnvValue] = useState('');
  const [newEnvEnvironment, setNewEnvEnvironment] = useState<Environment>('development');
  
  const [newSecretKey, setNewSecretKey] = useState('');
  const [newSecretValue, setNewSecretValue] = useState('');

  const getEnvironmentBadgeClass = (env: Environment) => {
    switch (env) {
      case 'production':
        return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'development':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'shared':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
    }
  };

  const toggleValueVisibility = (id: string) => {
    const newVisible = new Set(visibleValues);
    if (newVisible.has(id)) {
      newVisible.delete(id);
    } else {
      newVisible.add(id);
    }
    setVisibleValues(newVisible);
  };

  const maskValue = (value: string) => {
    return '•'.repeat(Math.min(value.length, 24));
  };

  const getDuplicateKeyWarning = (key: string, currentId?: string) => {
    const matches = envVariables.filter(v => v.key === key && v.id !== currentId);
    if (matches.length > 0) {
      const envs = matches.map(m => m.environment).join(', ');
      return `This key is also used in: ${envs}`;
    }
    return null;
  };

  const handleAddEnvVariable = () => {
    if (!newEnvKey.trim() || !newEnvValue.trim()) return;
    
    if (editingEnv) {
      setEnvVariables(vars => vars.map(v => 
        v.id === editingEnv.id 
          ? { ...v, key: newEnvKey, value: newEnvValue, environment: newEnvEnvironment }
          : v
      ));
    } else {
      const newVar: EnvVariable = {
        id: Date.now().toString(),
        key: newEnvKey,
        value: newEnvValue,
        environment: newEnvEnvironment,
      };
      setEnvVariables([...envVariables, newVar]);
    }
    
    resetEnvDialog();
  };

  const handleAddSecret = () => {
    if (!newSecretKey.trim() || !newSecretValue.trim()) return;
    
    if (editingSecret) {
      setSecrets(secs => secs.map(s => 
        s.id === editingSecret.id 
          ? { ...s }
          : s
      ));
    } else {
      const newSec: Secret = {
        id: Date.now().toString(),
        key: newSecretKey,
        createdAt: new Date().toISOString().split('T')[0],
      };
      setSecrets([...secrets, newSec]);
    }
    
    resetSecretDialog();
  };

  const handleEditEnv = (envVar: EnvVariable) => {
    setEditingEnv(envVar);
    setNewEnvKey(envVar.key);
    setNewEnvValue(envVar.value);
    setNewEnvEnvironment(envVar.environment);
    setEnvDialogOpen(true);
  };

  const handleEditSecret = (secret: Secret) => {
    setEditingSecret(secret);
    setNewSecretKey(secret.key);
    setNewSecretValue('');
    setSecretDialogOpen(true);
  };

  const handleDeleteEnv = () => {
    if (deletingEnv) {
      setEnvVariables(vars => vars.filter(v => v.id !== deletingEnv.id));
      setDeletingEnv(null);
    }
    setDeleteDialogOpen(false);
  };

  const handleDeleteSecret = () => {
    if (deletingSecret) {
      setSecrets(secs => secs.filter(s => s.id !== deletingSecret.id));
      setDeletingSecret(null);
    }
    setDeleteSecretDialogOpen(false);
  };

  const resetEnvDialog = () => {
    setEnvDialogOpen(false);
    setEditingEnv(null);
    setNewEnvKey('');
    setNewEnvValue('');
    setNewEnvEnvironment('development');
  };

  const resetSecretDialog = () => {
    setSecretDialogOpen(false);
    setEditingSecret(null);
    setNewSecretKey('');
    setNewSecretValue('');
  };

  const duplicateWarning = newEnvKey ? getDuplicateKeyWarning(newEnvKey, editingEnv?.id) : null;

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Environment Variables & Secrets</h2>
          <p className="text-muted-foreground">Manage your application's configuration and sensitive data</p>
        </div>

        <Card className="bg-card/50 border-white/5">
          <CardHeader className="flex flex-row items-center justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5 text-primary" />
                Environment Variables
              </CardTitle>
              <CardDescription>
                Configure variables for different environments
              </CardDescription>
            </div>
            <Button 
              onClick={() => setEnvDialogOpen(true)}
              data-testid="button-add-variable"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Variable
            </Button>
          </CardHeader>
          <CardContent>
            {envVariables.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No environment variables configured
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="border-white/5 hover:bg-transparent">
                    <TableHead>Key</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Environment</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {envVariables.map((envVar) => {
                    const warning = getDuplicateKeyWarning(envVar.key, envVar.id);
                    return (
                      <TableRow 
                        key={envVar.id} 
                        className="border-white/5"
                        data-testid={`row-env-${envVar.id}`}
                      >
                        <TableCell className="font-mono text-sm">
                          <div className="flex items-center gap-2">
                            {envVar.key}
                            {warning && (
                              <span title={warning}>
                                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                              </span>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="font-mono text-sm">
                          <div className="flex items-center gap-2">
                            <span className="text-muted-foreground" data-testid={`text-value-${envVar.id}`}>
                              {visibleValues.has(envVar.id) ? envVar.value : maskValue(envVar.value)}
                            </span>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7"
                              onClick={() => toggleValueVisibility(envVar.id)}
                              data-testid={`button-toggle-visibility-${envVar.id}`}
                            >
                              {visibleValues.has(envVar.id) ? (
                                <EyeOff className="h-4 w-4" />
                              ) : (
                                <Eye className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge 
                            variant="outline" 
                            className={getEnvironmentBadgeClass(envVar.environment)}
                            data-testid={`badge-env-${envVar.id}`}
                          >
                            {envVar.environment}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8"
                              onClick={() => handleEditEnv(envVar)}
                              data-testid={`button-edit-env-${envVar.id}`}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-red-400 hover:text-red-400"
                              onClick={() => {
                                setDeletingEnv(envVar);
                                setDeleteDialogOpen(true);
                              }}
                              data-testid={`button-delete-env-${envVar.id}`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-white/5">
          <CardHeader className="flex flex-row items-center justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Lock className="h-5 w-5 text-primary" />
                Secrets
              </CardTitle>
              <CardDescription>
                Encrypted secrets for sensitive data
              </CardDescription>
            </div>
            <Button 
              onClick={() => setSecretDialogOpen(true)}
              data-testid="button-add-secret"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Secret
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert className="border-amber-500/20 bg-amber-500/5">
              <ShieldAlert className="h-4 w-4 text-amber-500" />
              <AlertDescription className="text-amber-200/80">
                Secrets are encrypted and cannot be viewed after creation. You can only update or delete them.
              </AlertDescription>
            </Alert>
            
            {secrets.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No secrets configured
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="border-white/5 hover:bg-transparent">
                    <TableHead>Key</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {secrets.map((secret) => (
                    <TableRow 
                      key={secret.id} 
                      className="border-white/5"
                      data-testid={`row-secret-${secret.id}`}
                    >
                      <TableCell className="font-mono text-sm">
                        <div className="flex items-center gap-2">
                          <Lock className="h-4 w-4 text-muted-foreground" />
                          {secret.key}
                        </div>
                      </TableCell>
                      <TableCell className="font-mono text-sm text-muted-foreground">
                        {'•'.repeat(16)}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {secret.createdAt}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleEditSecret(secret)}
                            data-testid={`button-edit-secret-${secret.id}`}
                          >
                            <Pencil className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-red-400 hover:text-red-400"
                            onClick={() => {
                              setDeletingSecret(secret);
                              setDeleteSecretDialogOpen(true);
                            }}
                            data-testid={`button-delete-secret-${secret.id}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Dialog open={envDialogOpen} onOpenChange={(open) => !open && resetEnvDialog()}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>
                {editingEnv ? 'Edit Environment Variable' : 'Add Environment Variable'}
              </DialogTitle>
              <DialogDescription>
                {editingEnv 
                  ? 'Update the environment variable details below.'
                  : 'Add a new environment variable to your project.'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="env-key">Key</Label>
                <Input
                  id="env-key"
                  placeholder="e.g., API_URL"
                  value={newEnvKey}
                  onChange={(e) => setNewEnvKey(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ''))}
                  className="font-mono"
                  data-testid="input-env-key"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="env-value">Value</Label>
                <Input
                  id="env-value"
                  placeholder="Enter value"
                  value={newEnvValue}
                  onChange={(e) => setNewEnvValue(e.target.value)}
                  data-testid="input-env-value"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="env-environment">Environment</Label>
                <Select 
                  value={newEnvEnvironment} 
                  onValueChange={(val) => setNewEnvEnvironment(val as Environment)}
                >
                  <SelectTrigger data-testid="select-env-environment">
                    <SelectValue placeholder="Select environment" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="development">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-blue-500" />
                        Development
                      </div>
                    </SelectItem>
                    <SelectItem value="production">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-green-500" />
                        Production
                      </div>
                    </SelectItem>
                    <SelectItem value="shared">
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-2 rounded-full bg-purple-500" />
                        Shared
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {duplicateWarning && (
                <Alert className="border-yellow-500/20 bg-yellow-500/5">
                  <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  <AlertDescription className="text-yellow-200/80">
                    {duplicateWarning}
                  </AlertDescription>
                </Alert>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={resetEnvDialog} data-testid="button-cancel-env">
                Cancel
              </Button>
              <Button 
                onClick={handleAddEnvVariable}
                disabled={!newEnvKey.trim() || !newEnvValue.trim()}
                data-testid="button-save-env"
              >
                {editingEnv ? 'Update' : 'Add Variable'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={secretDialogOpen} onOpenChange={(open) => !open && resetSecretDialog()}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>
                {editingSecret ? 'Update Secret' : 'Add Secret'}
              </DialogTitle>
              <DialogDescription>
                {editingSecret 
                  ? 'Enter a new value for this secret. The key cannot be changed.'
                  : 'Add a new encrypted secret to your project.'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="secret-key">Key</Label>
                <Input
                  id="secret-key"
                  placeholder="e.g., STRIPE_SECRET_KEY"
                  value={newSecretKey}
                  onChange={(e) => setNewSecretKey(e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, ''))}
                  className="font-mono"
                  disabled={!!editingSecret}
                  data-testid="input-secret-key"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="secret-value">Value</Label>
                <Input
                  id="secret-value"
                  type="password"
                  placeholder={editingSecret ? 'Enter new value' : 'Enter secret value'}
                  value={newSecretValue}
                  onChange={(e) => setNewSecretValue(e.target.value)}
                  data-testid="input-secret-value"
                />
              </div>
              <Alert className="border-blue-500/20 bg-blue-500/5">
                <Lock className="h-4 w-4 text-blue-500" />
                <AlertDescription className="text-blue-200/80">
                  This value will be encrypted and cannot be viewed after saving.
                </AlertDescription>
              </Alert>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={resetSecretDialog} data-testid="button-cancel-secret">
                Cancel
              </Button>
              <Button 
                onClick={handleAddSecret}
                disabled={!newSecretKey.trim() || !newSecretValue.trim()}
                data-testid="button-save-secret"
              >
                {editingSecret ? 'Update Secret' : 'Add Secret'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Environment Variable</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete <span className="font-mono font-semibold">{deletingEnv?.key}</span>? 
                This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel data-testid="button-cancel-delete-env">Cancel</AlertDialogCancel>
              <AlertDialogAction 
                onClick={handleDeleteEnv}
                className="bg-red-600 hover:bg-red-700"
                data-testid="button-confirm-delete-env"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <AlertDialog open={deleteSecretDialogOpen} onOpenChange={setDeleteSecretDialogOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Secret</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete <span className="font-mono font-semibold">{deletingSecret?.key}</span>? 
                This action cannot be undone and the secret value will be permanently lost.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel data-testid="button-cancel-delete-secret">Cancel</AlertDialogCancel>
              <AlertDialogAction 
                onClick={handleDeleteSecret}
                className="bg-red-600 hover:bg-red-700"
                data-testid="button-confirm-delete-secret"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </DashboardLayout>
  );
}
