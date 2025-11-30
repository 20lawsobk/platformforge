import { useState } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { Cpu, Rocket, Zap, Code, Cloud, Check, ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";

const steps = [
  {
    id: 1,
    title: "Welcome to Platform Forge",
    description: "The AI-powered platform that transforms your code into production-ready infrastructure.",
    icon: Sparkles,
  },
  {
    id: 2,
    title: "How It Works",
    description: "Simply provide your code via GitHub URL or file upload, and we'll analyze it to generate optimized infrastructure configurations.",
    icon: Code,
  },
  {
    id: 3,
    title: "What You'll Get",
    description: "Terraform configs, Kubernetes manifests, Dockerfiles, auto-scaling rules, and security best practices - all tailored to your codebase.",
    icon: Cloud,
  },
];

const features = [
  { icon: Zap, label: "AI-Powered Analysis" },
  { icon: Cloud, label: "Multi-Cloud Support" },
  { icon: Rocket, label: "One-Click Deploy" },
];

export default function Onboarding() {
  const [currentStep, setCurrentStep] = useState(0);
  const [, setLocation] = useLocation();
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const completeOnboarding = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/auth/onboarding', { 
        method: 'POST',
        credentials: 'include',
      });
      if (!response.ok) throw new Error('Failed to complete onboarding');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/auth/user'] });
      setLocation('/dashboard');
    },
  });

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      completeOnboarding.mutate();
    }
  };

  const handleSkip = () => {
    completeOnboarding.mutate();
  };

  const currentStepData = steps[currentStep];
  const StepIcon = currentStepData.icon;

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]" />
      
      <motion.div
        key={currentStep}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
        className="w-full max-w-2xl relative z-10"
      >
        <Card className="p-8 md:p-12">
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-2 mb-6">
              <div className="bg-primary/10 p-2 rounded-md">
                <Cpu className="h-6 w-6 text-primary" />
              </div>
              <span className="text-xl font-bold tracking-tight font-mono">
                Platform<span className="text-primary">Forge</span>
              </span>
            </div>

            {user && (
              <Badge variant="outline" className="mb-6" data-testid="badge-welcome">
                Welcome, {user.firstName || user.email || 'Builder'}
              </Badge>
            )}

            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-6">
              <StepIcon className="h-10 w-10 text-primary" />
            </div>

            <h1 className="text-2xl md:text-3xl font-bold mb-4" data-testid="text-step-title">
              {currentStepData.title}
            </h1>
            <p className="text-muted-foreground text-lg max-w-md mx-auto" data-testid="text-step-description">
              {currentStepData.description}
            </p>
          </div>

          {currentStep === 2 && (
            <div className="grid grid-cols-3 gap-4 mb-8">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex flex-col items-center gap-2 p-4 rounded-lg bg-muted/50"
                >
                  <feature.icon className="h-6 w-6 text-primary" />
                  <span className="text-sm font-medium text-center">{feature.label}</span>
                </motion.div>
              ))}
            </div>
          )}

          <div className="flex items-center justify-center gap-2 mb-8">
            {steps.map((_, index) => (
              <div
                key={index}
                className={`h-2 rounded-full transition-all ${
                  index === currentStep
                    ? "w-8 bg-primary"
                    : index < currentStep
                    ? "w-2 bg-primary/50"
                    : "w-2 bg-muted"
                }`}
              />
            ))}
          </div>

          <div className="flex items-center justify-between gap-4">
            <Button
              variant="ghost"
              onClick={handleSkip}
              disabled={completeOnboarding.isPending}
              data-testid="button-skip"
            >
              Skip
            </Button>

            <Button
              onClick={handleNext}
              disabled={completeOnboarding.isPending}
              className="gap-2"
              data-testid="button-next"
            >
              {currentStep === steps.length - 1 ? (
                <>
                  {completeOnboarding.isPending ? "Setting up..." : "Get Started"}
                  <Check className="h-4 w-4" />
                </>
              ) : (
                <>
                  Next
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}
