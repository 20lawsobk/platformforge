import { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Trash2, Plus, Save, Eye, EyeOff } from "lucide-react";

export default function Settings() {
  const [envVars, setEnvVars] = useState([
    { key: "DATABASE_URL", value: "postgres://user:pass@db.platform.net:5432/prod", visible: false },
    { key: "API_KEY", value: "sk_live_51Mz...", visible: false },
    { key: "NODE_ENV", value: "production", visible: true },
  ]);

  const toggleVisibility = (index: number) => {
    const newVars = [...envVars];
    newVars[index].visible = !newVars[index].visible;
    setEnvVars(newVars);
  };

  return (
    <DashboardLayout>
      <div className="space-y-8 max-w-4xl">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Project Settings</h2>
          <p className="text-muted-foreground">Configure your build settings, environment variables, and domains.</p>
        </div>

        <Card className="bg-card/50 border-white/5">
          <CardHeader>
            <CardTitle>Build Configuration</CardTitle>
            <CardDescription>Settings for how we build your project.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="framework">Framework Preset</Label>
              <Input id="framework" value="Next.js" disabled className="bg-white/5 border-white/10" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="grid gap-2">
                <Label htmlFor="buildCommand">Build Command</Label>
                <Input id="buildCommand" placeholder="npm run build" className="bg-background border-white/10 font-mono" />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="outputDir">Output Directory</Label>
                <Input id="outputDir" placeholder=".next" className="bg-background border-white/10 font-mono" />
              </div>
            </div>
            <div className="grid gap-2">
                <Label htmlFor="installCommand">Install Command</Label>
                <Input id="installCommand" placeholder="npm install" className="bg-background border-white/10 font-mono" />
            </div>
            <div className="pt-2">
               <Button><Save className="h-4 w-4 mr-2" /> Save Configuration</Button>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-white/5">
          <CardHeader>
            <CardTitle>Environment Variables</CardTitle>
            <CardDescription>Secrets and configuration for your application environment.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-md border border-white/10 bg-background">
               {envVars.map((env, i) => (
                 <div key={i} className="flex items-center gap-2 p-3 border-b border-white/10 last:border-0">
                    <Input 
                      value={env.key} 
                      className="w-1/3 bg-transparent border-none font-mono text-primary focus-visible:ring-0 px-0" 
                      readOnly 
                    />
                    <div className="h-4 w-px bg-white/10" />
                    <div className="flex-1 relative">
                       <Input 
                         value={env.visible ? env.value : "••••••••••••••••"} 
                         className="bg-transparent border-none font-mono focus-visible:ring-0 px-0 w-full" 
                         readOnly 
                       />
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => toggleVisibility(i)}>
                       {env.visible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                    <Button variant="ghost" size="icon" className="text-red-400 hover:text-red-300 hover:bg-red-400/10">
                       <Trash2 className="h-4 w-4" />
                    </Button>
                 </div>
               ))}
               <div className="p-3 bg-white/5 flex gap-2">
                  <Input placeholder="KEY" className="w-1/3 bg-background border-white/10 font-mono" />
                  <Input placeholder="VALUE" className="flex-1 bg-background border-white/10 font-mono" />
                  <Button variant="secondary"><Plus className="h-4 w-4" /> Add</Button>
               </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-white/5 border-l-4 border-l-red-500">
           <CardHeader>
              <CardTitle className="text-red-500">Danger Zone</CardTitle>
              <CardDescription>Irreversible actions for your project.</CardDescription>
           </CardHeader>
           <CardContent>
              <div className="flex items-center justify-between">
                 <div>
                    <h4 className="font-medium">Delete Project</h4>
                    <p className="text-sm text-muted-foreground">Permanently remove this project and all of its deployments.</p>
                 </div>
                 <Button variant="destructive">Delete Project</Button>
              </div>
           </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}