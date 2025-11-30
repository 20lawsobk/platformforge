import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { format } from 'date-fns';
import {
  Database,
  HardDrive,
  Plus,
  Trash2,
  Edit,
  ChevronDown,
  ChevronRight,
  Key,
  FileText,
  Globe,
  Lock,
  Upload,
  Loader2
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Checkbox } from '@/components/ui/checkbox';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import DashboardLayout from '@/components/DashboardLayout';
import { apiRequest, queryClient } from '@/lib/queryClient';
import type { KvNamespace, KvEntry, ObjectBucket, StorageObject } from '@shared/schema';

const namespaceFormSchema = z.object({
  name: z.string().min(1, 'Name is required').max(50, 'Name must be 50 characters or less'),
  description: z.string().max(200, 'Description must be 200 characters or less').optional(),
});

const entryFormSchema = z.object({
  key: z.string().min(1, 'Key is required').max(256, 'Key must be 256 characters or less'),
  value: z.string().min(1, 'Value is required'),
});

const bucketFormSchema = z.object({
  name: z.string().min(1, 'Name is required').max(50, 'Name must be 50 characters or less'),
  description: z.string().max(200, 'Description must be 200 characters or less').optional(),
  isPublic: z.boolean().default(false),
});

const objectFormSchema = z.object({
  key: z.string().min(1, 'Key is required').max(256, 'Key must be 256 characters or less'),
  contentType: z.string().max(100, 'Content type must be 100 characters or less').optional(),
  size: z.number().min(0).default(0),
});

type NamespaceFormValues = z.infer<typeof namespaceFormSchema>;
type EntryFormValues = z.infer<typeof entryFormSchema>;
type BucketFormValues = z.infer<typeof bucketFormSchema>;
type ObjectFormValues = z.infer<typeof objectFormSchema>;

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function truncateValue(value: string, maxLength: number = 50): string {
  if (value.length <= maxLength) return value;
  return value.substring(0, maxLength) + '...';
}

function KVStoreTab() {
  const [expandedNamespace, setExpandedNamespace] = useState<string | null>(null);
  const [createNamespaceOpen, setCreateNamespaceOpen] = useState(false);
  const [createEntryOpen, setCreateEntryOpen] = useState(false);
  const [editEntryOpen, setEditEntryOpen] = useState(false);
  const [deleteNamespaceOpen, setDeleteNamespaceOpen] = useState(false);
  const [deleteEntryOpen, setDeleteEntryOpen] = useState(false);
  const [selectedNamespace, setSelectedNamespace] = useState<KvNamespace | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<KvEntry | null>(null);

  const { data: namespaces = [], isLoading: namespacesLoading } = useQuery<KvNamespace[]>({
    queryKey: ['/api/storage/kv/namespaces'],
  });

  const { data: entries = [], isLoading: entriesLoading } = useQuery<KvEntry[]>({
    queryKey: ['/api/storage/kv/namespaces', expandedNamespace, 'entries'],
    enabled: !!expandedNamespace,
  });

  const namespaceForm = useForm<NamespaceFormValues>({
    resolver: zodResolver(namespaceFormSchema),
    defaultValues: { name: '', description: '' },
  });

  const entryForm = useForm<EntryFormValues>({
    resolver: zodResolver(entryFormSchema),
    defaultValues: { key: '', value: '' },
  });

  const editEntryForm = useForm<EntryFormValues>({
    resolver: zodResolver(entryFormSchema),
    defaultValues: { key: '', value: '' },
  });

  const createNamespaceMutation = useMutation({
    mutationFn: async (data: NamespaceFormValues) => {
      const response = await apiRequest('POST', '/api/storage/kv/namespaces', data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces'] });
      setCreateNamespaceOpen(false);
      namespaceForm.reset();
    },
  });

  const deleteNamespaceMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest('DELETE', `/api/storage/kv/namespaces/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces'] });
      setDeleteNamespaceOpen(false);
      setSelectedNamespace(null);
      if (expandedNamespace === selectedNamespace?.id) {
        setExpandedNamespace(null);
      }
    },
  });

  const createEntryMutation = useMutation({
    mutationFn: async (data: EntryFormValues) => {
      const response = await apiRequest('POST', `/api/storage/kv/namespaces/${expandedNamespace}/entries`, data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces', expandedNamespace, 'entries'] });
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces'] });
      setCreateEntryOpen(false);
      entryForm.reset();
    },
  });

  const updateEntryMutation = useMutation({
    mutationFn: async ({ id, value }: { id: string; value: string }) => {
      await apiRequest('PUT', `/api/storage/kv/entries/${id}`, { value });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces', expandedNamespace, 'entries'] });
      setEditEntryOpen(false);
      setSelectedEntry(null);
      editEntryForm.reset();
    },
  });

  const deleteEntryMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest('DELETE', `/api/storage/kv/entries/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces', expandedNamespace, 'entries'] });
      queryClient.invalidateQueries({ queryKey: ['/api/storage/kv/namespaces'] });
      setDeleteEntryOpen(false);
      setSelectedEntry(null);
    },
  });

  const handleNamespaceClick = (namespace: KvNamespace) => {
    setExpandedNamespace(expandedNamespace === namespace.id ? null : namespace.id);
  };

  const handleEditEntry = (entry: KvEntry) => {
    setSelectedEntry(entry);
    editEntryForm.reset({ key: entry.key, value: entry.value });
    setEditEntryOpen(true);
  };

  const handleDeleteNamespace = (namespace: KvNamespace) => {
    setSelectedNamespace(namespace);
    setDeleteNamespaceOpen(true);
  };

  const handleDeleteEntry = (entry: KvEntry) => {
    setSelectedEntry(entry);
    setDeleteEntryOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold">Key-Value Namespaces</h3>
          <p className="text-sm text-muted-foreground">Manage your key-value storage namespaces</p>
        </div>
        <Button onClick={() => setCreateNamespaceOpen(true)} data-testid="button-create-namespace">
          <Plus className="h-4 w-4 mr-2" /> Create Namespace
        </Button>
      </div>

      {namespacesLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : namespaces.length === 0 ? (
        <Card className="bg-card/50 border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Database className="h-12 w-12 text-muted-foreground mb-4" />
            <h4 className="text-lg font-medium mb-2">No namespaces yet</h4>
            <p className="text-sm text-muted-foreground mb-4 text-center">
              Create your first namespace to start storing key-value pairs
            </p>
            <Button onClick={() => setCreateNamespaceOpen(true)} data-testid="button-create-first-namespace">
              <Plus className="h-4 w-4 mr-2" /> Create Namespace
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {namespaces.map((namespace) => (
            <Card key={namespace.id} className="bg-card/50" data-testid={`card-namespace-${namespace.id}`}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <button
                    onClick={() => handleNamespaceClick(namespace)}
                    className="flex items-center gap-3 text-left"
                    data-testid={`button-expand-namespace-${namespace.id}`}
                  >
                    {expandedNamespace === namespace.id ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                    <div className="h-9 w-9 rounded-md bg-primary/10 flex items-center justify-center">
                      <Database className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-base" data-testid={`text-namespace-name-${namespace.id}`}>
                        {namespace.name}
                      </CardTitle>
                      {namespace.description && (
                        <p className="text-sm text-muted-foreground">{namespace.description}</p>
                      )}
                    </div>
                  </button>
                  <div className="flex items-center gap-3 flex-wrap">
                    <Badge variant="secondary" className="text-xs" data-testid={`badge-entry-count-${namespace.id}`}>
                      <Key className="h-3 w-3 mr-1" />
                      {entries.length || 0} entries
                    </Badge>
                    <span className="text-xs text-muted-foreground" data-testid={`text-namespace-created-${namespace.id}`}>
                      Created {namespace.createdAt ? format(new Date(namespace.createdAt), 'MMM d, yyyy') : 'N/A'}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteNamespace(namespace)}
                      data-testid={`button-delete-namespace-${namespace.id}`}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              {expandedNamespace === namespace.id && (
                <CardContent className="pt-0">
                  <div className="border-t pt-4">
                    <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
                      <h4 className="text-sm font-medium">Entries</h4>
                      <Button size="sm" onClick={() => setCreateEntryOpen(true)} data-testid="button-add-entry">
                        <Plus className="h-3 w-3 mr-1" /> Add Entry
                      </Button>
                    </div>
                    
                    {entriesLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                      </div>
                    ) : entries.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        No entries in this namespace
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Key</TableHead>
                            <TableHead>Value</TableHead>
                            <TableHead>Created</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {entries.map((entry) => (
                            <TableRow key={entry.id} data-testid={`row-entry-${entry.id}`}>
                              <TableCell className="font-mono text-sm" data-testid={`text-entry-key-${entry.id}`}>
                                {entry.key}
                              </TableCell>
                              <TableCell className="font-mono text-sm text-muted-foreground" data-testid={`text-entry-value-${entry.id}`}>
                                {truncateValue(entry.value)}
                              </TableCell>
                              <TableCell className="text-sm text-muted-foreground" data-testid={`text-entry-created-${entry.id}`}>
                                {entry.createdAt ? format(new Date(entry.createdAt), 'MMM d, yyyy') : 'N/A'}
                              </TableCell>
                              <TableCell className="text-right">
                                <div className="flex items-center justify-end gap-1">
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => handleEditEntry(entry)}
                                    data-testid={`button-edit-entry-${entry.id}`}
                                  >
                                    <Edit className="h-4 w-4" />
                                  </Button>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={() => handleDeleteEntry(entry)}
                                    data-testid={`button-delete-entry-${entry.id}`}
                                  >
                                    <Trash2 className="h-4 w-4 text-destructive" />
                                  </Button>
                                </div>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}

      <Dialog open={createNamespaceOpen} onOpenChange={setCreateNamespaceOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Namespace</DialogTitle>
            <DialogDescription>Create a new key-value namespace to store your data</DialogDescription>
          </DialogHeader>
          <Form {...namespaceForm}>
            <form onSubmit={namespaceForm.handleSubmit((data) => createNamespaceMutation.mutate(data))} className="space-y-4">
              <FormField
                control={namespaceForm.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input placeholder="my-namespace" {...field} data-testid="input-namespace-name" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={namespaceForm.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="Describe this namespace..." {...field} data-testid="input-namespace-description" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setCreateNamespaceOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createNamespaceMutation.isPending} data-testid="button-submit-namespace">
                  {createNamespaceMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Create
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <Dialog open={createEntryOpen} onOpenChange={setCreateEntryOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Entry</DialogTitle>
            <DialogDescription>Add a new key-value entry to this namespace</DialogDescription>
          </DialogHeader>
          <Form {...entryForm}>
            <form onSubmit={entryForm.handleSubmit((data) => createEntryMutation.mutate(data))} className="space-y-4">
              <FormField
                control={entryForm.control}
                name="key"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Key</FormLabel>
                    <FormControl>
                      <Input placeholder="my-key" {...field} data-testid="input-entry-key" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={entryForm.control}
                name="value"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Value</FormLabel>
                    <FormControl>
                      <Input placeholder="Enter value..." {...field} data-testid="input-entry-value" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setCreateEntryOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createEntryMutation.isPending} data-testid="button-submit-entry">
                  {createEntryMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Add Entry
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <Dialog open={editEntryOpen} onOpenChange={setEditEntryOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Entry</DialogTitle>
            <DialogDescription>Update the value for key: {selectedEntry?.key}</DialogDescription>
          </DialogHeader>
          <Form {...editEntryForm}>
            <form onSubmit={editEntryForm.handleSubmit((data) => selectedEntry && updateEntryMutation.mutate({ id: selectedEntry.id, value: data.value }))} className="space-y-4">
              <FormField
                control={editEntryForm.control}
                name="key"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Key</FormLabel>
                    <FormControl>
                      <Input {...field} disabled data-testid="input-edit-entry-key" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={editEntryForm.control}
                name="value"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Value</FormLabel>
                    <FormControl>
                      <Input placeholder="Enter value..." {...field} data-testid="input-edit-entry-value" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setEditEntryOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={updateEntryMutation.isPending} data-testid="button-update-entry">
                  {updateEntryMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Update
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteNamespaceOpen} onOpenChange={setDeleteNamespaceOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Namespace</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the namespace "{selectedNamespace?.name}"? This will also delete all entries within it. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete-namespace">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => selectedNamespace && deleteNamespaceMutation.mutate(selectedNamespace.id)}
              className="bg-destructive text-destructive-foreground"
              data-testid="button-confirm-delete-namespace"
            >
              {deleteNamespaceMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deleteEntryOpen} onOpenChange={setDeleteEntryOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Entry</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the entry "{selectedEntry?.key}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete-entry">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => selectedEntry && deleteEntryMutation.mutate(selectedEntry.id)}
              className="bg-destructive text-destructive-foreground"
              data-testid="button-confirm-delete-entry"
            >
              {deleteEntryMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function ObjectStorageTab() {
  const [expandedBucket, setExpandedBucket] = useState<string | null>(null);
  const [createBucketOpen, setCreateBucketOpen] = useState(false);
  const [uploadObjectOpen, setUploadObjectOpen] = useState(false);
  const [deleteBucketOpen, setDeleteBucketOpen] = useState(false);
  const [deleteObjectOpen, setDeleteObjectOpen] = useState(false);
  const [selectedBucket, setSelectedBucket] = useState<ObjectBucket | null>(null);
  const [selectedObject, setSelectedObject] = useState<StorageObject | null>(null);

  const { data: buckets = [], isLoading: bucketsLoading } = useQuery<ObjectBucket[]>({
    queryKey: ['/api/storage/buckets'],
  });

  const { data: objects = [], isLoading: objectsLoading } = useQuery<StorageObject[]>({
    queryKey: ['/api/storage/buckets', expandedBucket, 'objects'],
    enabled: !!expandedBucket,
  });

  const bucketForm = useForm<BucketFormValues>({
    resolver: zodResolver(bucketFormSchema),
    defaultValues: { name: '', description: '', isPublic: false },
  });

  const objectForm = useForm<ObjectFormValues>({
    resolver: zodResolver(objectFormSchema),
    defaultValues: { key: '', contentType: '', size: 0 },
  });

  const createBucketMutation = useMutation({
    mutationFn: async (data: BucketFormValues) => {
      const response = await apiRequest('POST', '/api/storage/buckets', data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets'] });
      setCreateBucketOpen(false);
      bucketForm.reset();
    },
  });

  const deleteBucketMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest('DELETE', `/api/storage/buckets/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets'] });
      setDeleteBucketOpen(false);
      setSelectedBucket(null);
      if (expandedBucket === selectedBucket?.id) {
        setExpandedBucket(null);
      }
    },
  });

  const uploadObjectMutation = useMutation({
    mutationFn: async (data: ObjectFormValues) => {
      const response = await apiRequest('POST', `/api/storage/buckets/${expandedBucket}/objects`, data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets', expandedBucket, 'objects'] });
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets'] });
      setUploadObjectOpen(false);
      objectForm.reset();
    },
  });

  const deleteObjectMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest('DELETE', `/api/storage/objects/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets', expandedBucket, 'objects'] });
      queryClient.invalidateQueries({ queryKey: ['/api/storage/buckets'] });
      setDeleteObjectOpen(false);
      setSelectedObject(null);
    },
  });

  const handleBucketClick = (bucket: ObjectBucket) => {
    setExpandedBucket(expandedBucket === bucket.id ? null : bucket.id);
  };

  const handleDeleteBucket = (bucket: ObjectBucket) => {
    setSelectedBucket(bucket);
    setDeleteBucketOpen(true);
  };

  const handleDeleteObject = (obj: StorageObject) => {
    setSelectedObject(obj);
    setDeleteObjectOpen(true);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-lg font-semibold">Object Storage Buckets</h3>
          <p className="text-sm text-muted-foreground">Manage your object storage buckets and files</p>
        </div>
        <Button onClick={() => setCreateBucketOpen(true)} data-testid="button-create-bucket">
          <Plus className="h-4 w-4 mr-2" /> Create Bucket
        </Button>
      </div>

      {bucketsLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      ) : buckets.length === 0 ? (
        <Card className="bg-card/50 border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <HardDrive className="h-12 w-12 text-muted-foreground mb-4" />
            <h4 className="text-lg font-medium mb-2">No buckets yet</h4>
            <p className="text-sm text-muted-foreground mb-4 text-center">
              Create your first bucket to start storing objects
            </p>
            <Button onClick={() => setCreateBucketOpen(true)} data-testid="button-create-first-bucket">
              <Plus className="h-4 w-4 mr-2" /> Create Bucket
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {buckets.map((bucket) => (
            <Card key={bucket.id} className="bg-card/50" data-testid={`card-bucket-${bucket.id}`}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <button
                    onClick={() => handleBucketClick(bucket)}
                    className="flex items-center gap-3 text-left"
                    data-testid={`button-expand-bucket-${bucket.id}`}
                  >
                    {expandedBucket === bucket.id ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                    <div className="h-9 w-9 rounded-md bg-purple-500/10 flex items-center justify-center">
                      <HardDrive className="h-4 w-4 text-purple-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base" data-testid={`text-bucket-name-${bucket.id}`}>
                        {bucket.name}
                      </CardTitle>
                      {bucket.description && (
                        <p className="text-sm text-muted-foreground">{bucket.description}</p>
                      )}
                    </div>
                  </button>
                  <div className="flex items-center gap-3 flex-wrap">
                    <Badge
                      variant={bucket.isPublic ? 'default' : 'secondary'}
                      className="text-xs"
                      data-testid={`badge-bucket-visibility-${bucket.id}`}
                    >
                      {bucket.isPublic ? (
                        <>
                          <Globe className="h-3 w-3 mr-1" /> Public
                        </>
                      ) : (
                        <>
                          <Lock className="h-3 w-3 mr-1" /> Private
                        </>
                      )}
                    </Badge>
                    <Badge variant="secondary" className="text-xs" data-testid={`badge-object-count-${bucket.id}`}>
                      <FileText className="h-3 w-3 mr-1" />
                      {bucket.objectCount || 0} objects
                    </Badge>
                    <Badge variant="outline" className="text-xs" data-testid={`badge-bucket-size-${bucket.id}`}>
                      {formatBytes(bucket.totalSize || 0)}
                    </Badge>
                    <span className="text-xs text-muted-foreground" data-testid={`text-bucket-created-${bucket.id}`}>
                      Created {bucket.createdAt ? format(new Date(bucket.createdAt), 'MMM d, yyyy') : 'N/A'}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteBucket(bucket)}
                      data-testid={`button-delete-bucket-${bucket.id}`}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              {expandedBucket === bucket.id && (
                <CardContent className="pt-0">
                  <div className="border-t pt-4">
                    <div className="flex items-center justify-between mb-4 gap-4 flex-wrap">
                      <h4 className="text-sm font-medium">Objects</h4>
                      <Button size="sm" onClick={() => setUploadObjectOpen(true)} data-testid="button-upload-object">
                        <Upload className="h-3 w-3 mr-1" /> Upload Object
                      </Button>
                    </div>
                    
                    {objectsLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                      </div>
                    ) : objects.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        No objects in this bucket
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Key</TableHead>
                            <TableHead>Content Type</TableHead>
                            <TableHead>Size</TableHead>
                            <TableHead>Created</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {objects.map((obj) => (
                            <TableRow key={obj.id} data-testid={`row-object-${obj.id}`}>
                              <TableCell className="font-mono text-sm" data-testid={`text-object-key-${obj.id}`}>
                                {obj.key}
                              </TableCell>
                              <TableCell className="text-sm text-muted-foreground" data-testid={`text-object-type-${obj.id}`}>
                                {obj.contentType || 'N/A'}
                              </TableCell>
                              <TableCell className="text-sm text-muted-foreground" data-testid={`text-object-size-${obj.id}`}>
                                {formatBytes(obj.size || 0)}
                              </TableCell>
                              <TableCell className="text-sm text-muted-foreground" data-testid={`text-object-created-${obj.id}`}>
                                {obj.createdAt ? format(new Date(obj.createdAt), 'MMM d, yyyy') : 'N/A'}
                              </TableCell>
                              <TableCell className="text-right">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleDeleteObject(obj)}
                                  data-testid={`button-delete-object-${obj.id}`}
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </div>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      )}

      <Dialog open={createBucketOpen} onOpenChange={setCreateBucketOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Bucket</DialogTitle>
            <DialogDescription>Create a new bucket to store your objects</DialogDescription>
          </DialogHeader>
          <Form {...bucketForm}>
            <form onSubmit={bucketForm.handleSubmit((data) => createBucketMutation.mutate(data))} className="space-y-4">
              <FormField
                control={bucketForm.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Name</FormLabel>
                    <FormControl>
                      <Input placeholder="my-bucket" {...field} data-testid="input-bucket-name" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={bucketForm.control}
                name="description"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Description (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="Describe this bucket..." {...field} data-testid="input-bucket-description" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={bucketForm.control}
                name="isPublic"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center gap-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                        data-testid="checkbox-bucket-public"
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>Make bucket public</FormLabel>
                    </div>
                  </FormItem>
                )}
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setCreateBucketOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={createBucketMutation.isPending} data-testid="button-submit-bucket">
                  {createBucketMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Create
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <Dialog open={uploadObjectOpen} onOpenChange={setUploadObjectOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Upload Object</DialogTitle>
            <DialogDescription>Add object metadata to this bucket</DialogDescription>
          </DialogHeader>
          <Form {...objectForm}>
            <form onSubmit={objectForm.handleSubmit((data) => uploadObjectMutation.mutate(data))} className="space-y-4">
              <FormField
                control={objectForm.control}
                name="key"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Key</FormLabel>
                    <FormControl>
                      <Input placeholder="path/to/file.txt" {...field} data-testid="input-object-key" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={objectForm.control}
                name="contentType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Content Type (optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="application/json" {...field} data-testid="input-object-content-type" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={objectForm.control}
                name="size"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Size (bytes)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        placeholder="0"
                        {...field}
                        onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                        data-testid="input-object-size"
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <DialogFooter>
                <Button type="button" variant="outline" onClick={() => setUploadObjectOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={uploadObjectMutation.isPending} data-testid="button-submit-object">
                  {uploadObjectMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Upload
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteBucketOpen} onOpenChange={setDeleteBucketOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Bucket</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the bucket "{selectedBucket?.name}"? This will also delete all objects within it. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete-bucket">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => selectedBucket && deleteBucketMutation.mutate(selectedBucket.id)}
              className="bg-destructive text-destructive-foreground"
              data-testid="button-confirm-delete-bucket"
            >
              {deleteBucketMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={deleteObjectOpen} onOpenChange={setDeleteObjectOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Object</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete the object "{selectedObject?.key}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel data-testid="button-cancel-delete-object">Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => selectedObject && deleteObjectMutation.mutate(selectedObject.id)}
              className="bg-destructive text-destructive-foreground"
              data-testid="button-confirm-delete-object"
            >
              {deleteObjectMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default function Storage() {
  return (
    <DashboardLayout>
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Storage</h2>
          <p className="text-muted-foreground">Manage your key-value store and object storage</p>
        </div>

        <Tabs defaultValue="kv" className="space-y-6">
          <TabsList data-testid="tabs-storage">
            <TabsTrigger value="kv" data-testid="tab-kv-store">
              <Database className="h-4 w-4 mr-2" />
              Key-Value Store
            </TabsTrigger>
            <TabsTrigger value="objects" data-testid="tab-object-storage">
              <HardDrive className="h-4 w-4 mr-2" />
              Object Storage
            </TabsTrigger>
          </TabsList>

          <TabsContent value="kv">
            <KVStoreTab />
          </TabsContent>

          <TabsContent value="objects">
            <ObjectStorageTab />
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
}
