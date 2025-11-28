import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { ArrowRight, Github, Code2, Zap, Globe, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import Layout from "@/components/Layout";
import bgImage from "@assets/generated_images/cybernetic_schematic_background.png";

export default function Home() {
  const [input, setInput] = useState("");
  const [, setLocation] = useLocation();

  const handleIgnite = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      setLocation(`/builder?source=${encodeURIComponent(input)}`);
    }
  };

  return (
    <Layout>
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-background">
        {/* Background Asset */}
        <div className="absolute inset-0 z-0">
           {/* Grid Pattern Overlay */}
          <div className="absolute inset-0 bg-grid-pattern opacity-[0.03] z-10 pointer-events-none" />
          
          <img 
            src={bgImage} 
            alt="Schematic Background" 
            className="w-full h-full object-cover opacity-20 mix-blend-screen"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background/90 via-background/40 to-background" />
        </div>

        <div className="container relative z-10 px-4 text-center max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-mono font-bold mb-8 shadow-[0_0_15px_-3px_var(--color-primary)]"
            >
              <Zap className="h-3 w-3" />
              <span>V2.0 NOW LIVE: MULTI-REGION DEPLOYMENT</span>
            </motion.div>
            
            <h1 className="text-5xl md:text-7xl font-bold tracking-tighter mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              AI-Architected <br className="hidden md:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-cyan-400 to-blue-500">Platforms.</span>
            </h1>
            
            <p className="text-lg md:text-xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
              Don't just build apps. Generate enterprise-grade infrastructure.
              From scripts to auto-scaling Kubernetes clusters in one click.
            </p>

            <form onSubmit={handleIgnite} className="flex flex-col sm:flex-row gap-4 max-w-xl mx-auto p-2 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm shadow-2xl">
              <div className="relative flex-1">
                <Code2 className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
                <Input 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="https://github.com/user/repo or ./script.js" 
                  className="pl-10 h-12 bg-transparent border-none focus-visible:ring-0 text-base font-mono placeholder:text-muted-foreground/50"
                />
              </div>
              <Button type="submit" size="lg" className="h-12 px-8 bg-primary text-primary-foreground hover:bg-primary/90 font-bold transition-all hover:scale-105 hover:shadow-[0_0_20px_-5px_var(--color-primary)]">
                Ignite <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </form>

            <div className="mt-10 flex items-center justify-center gap-8 text-sm text-muted-foreground/60 font-mono">
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Github className="h-4 w-4" /> GitHub Supported</span>
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Code2 className="h-4 w-4" /> Python / Node / Go</span>
              <span className="flex items-center gap-2 hover:text-primary transition-colors cursor-default"><Globe className="h-4 w-4" /> Edge Ready</span>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="py-24 bg-secondary/30 border-t border-white/5 relative overflow-hidden">
         <div className="absolute inset-0 bg-grid-pattern opacity-[0.02] pointer-events-none" />
        <div className="container px-4 md:px-8 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-primary/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <Globe className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Global Edge Network</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Your platforms are automatically replicated across 35+ regions globally. sub-50ms latency for everyone.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-purple-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4 group-hover:bg-purple-500/20 transition-colors">
                <Zap className="h-6 w-6 text-purple-400" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Instant Scaling</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                From 0 to 10 million requests. Our serverless infrastructure handles the load so you don't have to.
              </p>
            </Card>

            <Card className="p-6 bg-card/50 border-white/5 backdrop-blur-sm hover:border-green-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="h-12 w-12 rounded-lg bg-green-500/10 flex items-center justify-center mb-4 group-hover:bg-green-500/20 transition-colors">
                <Shield className="h-6 w-6 text-green-400" />
              </div>
              <h3 className="text-xl font-bold mb-2 font-sans">Enterprise Grade</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                SOC2 Type II compliant. Automated DDoS protection, WAF, and encrypted secrets management built-in.
              </p>
            </Card>
          </div>
        </div>
      </section>
    </Layout>
  );
}