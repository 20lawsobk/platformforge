import { motion } from "framer-motion";
import {
  Users,
  Lightbulb,
  Shield,
  Code2,
  Heart,
  Rocket,
  Building2,
  Target,
  Sparkles,
  ArrowRight,
  Github,
  Linkedin,
  Twitter
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface TeamMember {
  name: string;
  role: string;
  bio: string;
  initials: string;
}

interface Value {
  icon: React.ElementType;
  title: string;
  description: string;
}

const teamMembers: TeamMember[] = [
  {
    name: "Alex Chen",
    role: "CEO & Co-Founder",
    bio: "Former infrastructure lead at a major cloud provider. Passionate about making deployment accessible to all developers.",
    initials: "AC"
  },
  {
    name: "Sarah Mitchell",
    role: "CTO & Co-Founder",
    bio: "15+ years in distributed systems. Previously built scalable platforms at leading tech companies.",
    initials: "SM"
  },
  {
    name: "Marcus Rodriguez",
    role: "VP of Engineering",
    bio: "Expert in AI/ML infrastructure. Led teams building real-time data processing systems.",
    initials: "MR"
  },
  {
    name: "Emily Watson",
    role: "Head of Product",
    bio: "Product leader focused on developer experience. Believes in building tools developers actually love.",
    initials: "EW"
  },
  {
    name: "David Park",
    role: "Lead Platform Engineer",
    bio: "Kubernetes contributor and cloud-native advocate. Loves solving complex infrastructure challenges.",
    initials: "DP"
  },
  {
    name: "Lisa Thompson",
    role: "Head of Developer Relations",
    bio: "Community builder and educator. Dedicated to helping developers succeed with modern tools.",
    initials: "LT"
  }
];

const values: Value[] = [
  {
    icon: Lightbulb,
    title: "Innovation",
    description: "We push boundaries with AI-powered infrastructure generation, constantly exploring new ways to simplify complex deployments."
  },
  {
    icon: Shield,
    title: "Security",
    description: "Security is not an afterthought. We build enterprise-grade protection into every layer of our platform from day one."
  },
  {
    icon: Code2,
    title: "Developer Experience",
    description: "Every feature is designed with developers in mind. We obsess over making infrastructure feel effortless."
  },
  {
    icon: Heart,
    title: "Open Source",
    description: "We believe in giving back. Our core tools are open source, and we actively contribute to the communities we depend on."
  }
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5
    }
  }
};

export default function About() {
  return (
    <Layout>
      <div className="min-h-screen bg-background">
        <section className="py-20 border-b border-border relative overflow-hidden">
          <div className="absolute inset-0 bg-grid-pattern opacity-[0.02]" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-about">
                <Building2 className="h-3 w-3 mr-1" /> About Us
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-about-title">
                Building the Future of<br />
                <span className="text-primary">Infrastructure</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-about-subtitle">
                We're on a mission to make infrastructure deployment as simple as writing code. 
                Our AI-powered platform transforms how teams build and scale applications.
              </p>
            </motion.div>
          </div>
        </section>

        <section className="py-20">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="grid md:grid-cols-2 gap-12 items-center"
            >
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <Rocket className="h-5 w-5 text-primary" />
                  <h2 className="text-2xl font-bold" data-testid="text-story-title">Our Story</h2>
                </div>
                <div className="space-y-4 text-muted-foreground" data-testid="text-story-content">
                  <p>
                    PlatformBuilder was founded in 2023 by a team of infrastructure engineers who were 
                    frustrated with the complexity of modern cloud deployments. We saw teams spending 
                    months configuring Kubernetes, wrestling with Terraform, and debugging deployment 
                    pipelines instead of building products.
                  </p>
                  <p>
                    We believed there had to be a better way. What if AI could understand your code 
                    and automatically generate production-ready infrastructure? What if deploying to 
                    the cloud was as simple as pushing to GitHub?
                  </p>
                  <p>
                    Today, PlatformBuilder powers thousands of applications worldwide, from startups 
                    to Fortune 500 companies. Our platform has helped teams reduce deployment time 
                    by 90% while maintaining enterprise-grade security and reliability.
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <Card className="p-6 text-center bg-card border-border" data-testid="stat-developers">
                  <div className="text-4xl font-bold text-primary mb-2">50K+</div>
                  <div className="text-sm text-muted-foreground">Developers</div>
                </Card>
                <Card className="p-6 text-center bg-card border-border" data-testid="stat-deployments">
                  <div className="text-4xl font-bold text-primary mb-2">1M+</div>
                  <div className="text-sm text-muted-foreground">Deployments</div>
                </Card>
                <Card className="p-6 text-center bg-card border-border" data-testid="stat-countries">
                  <div className="text-4xl font-bold text-primary mb-2">120+</div>
                  <div className="text-sm text-muted-foreground">Countries</div>
                </Card>
                <Card className="p-6 text-center bg-card border-border" data-testid="stat-uptime">
                  <div className="text-4xl font-bold text-primary mb-2">99.99%</div>
                  <div className="text-sm text-muted-foreground">Uptime</div>
                </Card>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-20 bg-card/30 border-y border-border">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="text-center mb-12"
            >
              <div className="flex items-center justify-center gap-2 mb-4">
                <Target className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-values-title">Our Values</h2>
              </div>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-values-subtitle">
                These core principles guide everything we do, from product decisions to how we work together.
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
              {values.map((value, index) => (
                <motion.div key={value.title} variants={itemVariants}>
                  <Card
                    className="p-6 bg-card border-border h-full text-center"
                    data-testid={`card-value-${index}`}
                  >
                    <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                      <value.icon className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="font-semibold text-lg mb-2" data-testid={`text-value-title-${index}`}>
                      {value.title}
                    </h3>
                    <p className="text-sm text-muted-foreground" data-testid={`text-value-description-${index}`}>
                      {value.description}
                    </p>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        <section className="py-20">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
              className="text-center mb-12"
            >
              <div className="flex items-center justify-center gap-2 mb-4">
                <Users className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-team-title">Meet Our Team</h2>
              </div>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-team-subtitle">
                We're a diverse team of engineers, designers, and operators united by our passion for developer tools.
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {teamMembers.map((member, index) => (
                <motion.div key={member.name} variants={itemVariants}>
                  <Card
                    className="p-6 bg-card border-border h-full"
                    data-testid={`card-team-member-${index}`}
                  >
                    <div className="flex items-start gap-4">
                      <Avatar className="h-14 w-14">
                        <AvatarFallback className="bg-primary/10 text-primary font-semibold">
                          {member.initials}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold" data-testid={`text-member-name-${index}`}>
                          {member.name}
                        </h3>
                        <p className="text-sm text-primary mb-2" data-testid={`text-member-role-${index}`}>
                          {member.role}
                        </p>
                        <p className="text-sm text-muted-foreground" data-testid={`text-member-bio-${index}`}>
                          {member.bio}
                        </p>
                        <div className="flex items-center gap-2 mt-3">
                          <Button variant="ghost" size="icon" className="h-8 w-8" data-testid={`button-member-twitter-${index}`}>
                            <Twitter className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8" data-testid={`button-member-linkedin-${index}`}>
                            <Linkedin className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8" data-testid={`button-member-github-${index}`}>
                            <Github className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </section>

        <section className="py-20 border-t border-border relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-cta-title">
                Join Our Journey
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Whether you want to build with us or for us, we'd love to have you along for the ride.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link href="/careers">
                  <Button size="lg" className="font-bold" data-testid="button-view-careers">
                    <Users className="mr-2 h-4 w-4" /> View Open Positions
                  </Button>
                </Link>
                <Link href="/">
                  <Button size="lg" variant="outline" data-testid="button-start-building">
                    Start Building <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
