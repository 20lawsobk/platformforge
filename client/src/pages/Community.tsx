import { motion } from "framer-motion";
import { 
  Users,
  MessageSquare,
  Github,
  Twitter,
  Star,
  Heart,
  ExternalLink,
  ArrowRight,
  Sparkles,
  Code,
  Rocket,
  Globe,
  BookOpen,
  HelpCircle
} from "lucide-react";
import { SiDiscord, SiStackoverflow } from "react-icons/si";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import Layout from "@/components/Layout";

interface CommunityChannel {
  title: string;
  description: string;
  icon: React.ElementType;
  members: string;
  href: string;
  color: string;
}

interface FeaturedProject {
  title: string;
  description: string;
  author: string;
  stars: number;
  tags: string[];
}

interface Contributor {
  name: string;
  role: string;
  initials: string;
  contributions: number;
}

const communityChannels: CommunityChannel[] = [
  {
    title: "Discord Server",
    description: "Join our Discord for real-time discussions, support, and community events",
    icon: SiDiscord,
    members: "12,500+",
    href: "#discord",
    color: "from-indigo-500 to-purple-600"
  },
  {
    title: "GitHub Discussions",
    description: "Participate in technical discussions, feature requests, and RFC proposals",
    icon: Github,
    members: "8,200+",
    href: "#github",
    color: "from-gray-600 to-gray-800"
  },
  {
    title: "Stack Overflow",
    description: "Get answers to technical questions with the platform-forge tag",
    icon: SiStackoverflow,
    members: "5,800+",
    href: "#stackoverflow",
    color: "from-orange-500 to-orange-600"
  },
  {
    title: "Twitter / X",
    description: "Follow us for the latest updates, tips, and community highlights",
    icon: Twitter,
    members: "25,000+",
    href: "#twitter",
    color: "from-blue-400 to-blue-600"
  }
];

const featuredProjects: FeaturedProject[] = [
  {
    title: "forge-cli",
    description: "Command-line interface for Platform Forge with advanced deployment features",
    author: "community",
    stars: 1240,
    tags: ["CLI", "Go", "DevOps"]
  },
  {
    title: "forge-terraform-modules",
    description: "Collection of reusable Terraform modules for common infrastructure patterns",
    author: "community",
    stars: 890,
    tags: ["Terraform", "AWS", "GCP"]
  },
  {
    title: "forge-vscode",
    description: "VS Code extension for Platform Forge with IntelliSense and deployment integration",
    author: "community",
    stars: 2100,
    tags: ["VS Code", "TypeScript", "Extension"]
  },
  {
    title: "forge-github-action",
    description: "GitHub Action for seamless CI/CD integration with Platform Forge deployments",
    author: "community",
    stars: 560,
    tags: ["GitHub Actions", "CI/CD", "Automation"]
  }
];

const contributors: Contributor[] = [
  { name: "Alex Chen", role: "Core Maintainer", initials: "AC", contributions: 342 },
  { name: "Sarah Kim", role: "Documentation Lead", initials: "SK", contributions: 189 },
  { name: "Marcus Johnson", role: "Community Lead", initials: "MJ", contributions: 156 },
  { name: "Priya Sharma", role: "DevRel", initials: "PS", contributions: 134 },
  { name: "James Wilson", role: "Contributor", initials: "JW", contributions: 98 },
  { name: "Lisa Wang", role: "Contributor", initials: "LW", contributions: 87 },
  { name: "David Brown", role: "Contributor", initials: "DB", contributions: 76 },
  { name: "Emma Davis", role: "Contributor", initials: "ED", contributions: 65 }
];

export default function Community() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-community">
                <Users className="h-3 w-3 mr-1" /> Community
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-community-title">
                Join Our <span className="text-primary">Community</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-10" data-testid="text-community-subtitle">
                Connect with thousands of developers building the future of infrastructure. 
                Share knowledge, get help, and contribute to the ecosystem.
              </p>

              <div className="flex justify-center gap-8 flex-wrap">
                <div className="text-center" data-testid="stat-members">
                  <div className="text-4xl font-bold text-primary">50K+</div>
                  <div className="text-sm text-muted-foreground">Community Members</div>
                </div>
                <div className="text-center" data-testid="stat-projects">
                  <div className="text-4xl font-bold text-primary">2,500+</div>
                  <div className="text-sm text-muted-foreground">Open Source Projects</div>
                </div>
                <div className="text-center" data-testid="stat-contributors">
                  <div className="text-4xl font-bold text-primary">500+</div>
                  <div className="text-sm text-muted-foreground">Contributors</div>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="mb-10"
            >
              <div className="flex items-center gap-2 mb-2">
                <MessageSquare className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-channels-title">Connect With Us</h2>
              </div>
              <p className="text-muted-foreground">Find us on your favorite platform</p>
            </motion.div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {communityChannels.map((channel, index) => (
                <motion.div
                  key={channel.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 + index * 0.05 }}
                >
                  <a href={channel.href} data-testid={`link-channel-${index}`}>
                    <Card 
                      className="p-6 bg-card border-border h-full hover-elevate cursor-pointer group"
                      data-testid={`card-channel-${index}`}
                    >
                      <div className={`h-12 w-12 rounded-xl bg-gradient-to-br ${channel.color} flex items-center justify-center mb-4`}>
                        <channel.icon className="h-6 w-6 text-white" />
                      </div>
                      <h3 className="font-semibold mb-2 group-hover:text-primary transition-colors" data-testid={`text-channel-title-${index}`}>
                        {channel.title}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-4" data-testid={`text-channel-desc-${index}`}>
                        {channel.description}
                      </p>
                      <div className="flex items-center justify-between">
                        <Badge variant="secondary" className="text-xs" data-testid={`badge-members-${index}`}>
                          <Users className="h-3 w-3 mr-1" />
                          {channel.members}
                        </Badge>
                        <ExternalLink className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </Card>
                  </a>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 bg-card/30">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-10"
            >
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-projects-title">Featured Community Projects</h2>
              </div>
              <p className="text-muted-foreground">Open source projects built by our community</p>
            </motion.div>

            <div className="grid md:grid-cols-2 gap-6">
              {featuredProjects.map((project, index) => (
                <motion.div
                  key={project.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.05 }}
                >
                  <Card 
                    className="p-6 bg-card border-border h-full hover-elevate cursor-pointer group"
                    data-testid={`card-project-${index}`}
                  >
                    <div className="flex items-start justify-between gap-4 mb-4">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <Code className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <h3 className="font-semibold group-hover:text-primary transition-colors" data-testid={`text-project-title-${index}`}>
                            {project.title}
                          </h3>
                          <p className="text-xs text-muted-foreground">by {project.author}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <Star className="h-4 w-4" />
                        <span className="text-sm" data-testid={`text-stars-${index}`}>{project.stars.toLocaleString()}</span>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4" data-testid={`text-project-desc-${index}`}>
                      {project.description}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {project.tags.map((tag, tagIndex) => (
                        <Badge 
                          key={tag} 
                          variant="outline" 
                          className="text-xs"
                          data-testid={`badge-tag-${index}-${tagIndex}`}
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="container px-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-10"
            >
              <div className="flex items-center gap-2 mb-2">
                <Heart className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-contributors-title">Top Contributors</h2>
              </div>
              <p className="text-muted-foreground">The amazing people who make this community thrive</p>
            </motion.div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {contributors.map((contributor, index) => (
                <motion.div
                  key={contributor.name}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 + index * 0.05 }}
                >
                  <Card 
                    className="p-4 bg-card border-border text-center hover-elevate cursor-pointer"
                    data-testid={`card-contributor-${index}`}
                  >
                    <Avatar className="h-16 w-16 mx-auto mb-3">
                      <AvatarFallback className="bg-primary/10 text-primary text-lg font-semibold">
                        {contributor.initials}
                      </AvatarFallback>
                    </Avatar>
                    <h3 className="font-medium text-sm" data-testid={`text-contributor-name-${index}`}>
                      {contributor.name}
                    </h3>
                    <p className="text-xs text-muted-foreground mb-2" data-testid={`text-contributor-role-${index}`}>
                      {contributor.role}
                    </p>
                    <Badge variant="secondary" className="text-xs" data-testid={`badge-contributions-${index}`}>
                      {contributor.contributions} contributions
                    </Badge>
                  </Card>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 bg-card/30 border-t border-border">
          <div className="container px-4">
            <div className="grid md:grid-cols-3 gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <Card className="p-6 bg-card border-border h-full" data-testid="card-resource-docs">
                  <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                    <BookOpen className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold mb-2">Documentation</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Comprehensive guides and tutorials to help you get started
                  </p>
                  <Button variant="outline" size="sm" data-testid="button-view-docs">
                    View Docs <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <Card className="p-6 bg-card border-border h-full" data-testid="card-resource-contribute">
                  <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                    <Rocket className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold mb-2">Contribute</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Help us improve Platform Forge by contributing code or docs
                  </p>
                  <Button variant="outline" size="sm" data-testid="button-contribute">
                    Start Contributing <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                <Card className="p-6 bg-card border-border h-full" data-testid="card-resource-support">
                  <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                    <HelpCircle className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="font-semibold mb-2">Get Support</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Need help? Our community and support team are here for you
                  </p>
                  <Button variant="outline" size="sm" data-testid="button-get-support">
                    Get Help <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </Card>
              </motion.div>
            </div>
          </div>
        </section>

        <section className="py-20 border-t border-border relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <div className="h-14 w-14 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Globe className="h-7 w-7 text-primary" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-cta-title">
                Ready to Join?
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Become part of a global community of developers building amazing things with Platform Forge.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="font-bold" data-testid="button-join-discord-cta">
                  <SiDiscord className="h-5 w-5 mr-2" />
                  Join Discord
                </Button>
                <Button size="lg" variant="outline" data-testid="button-github-cta">
                  <Github className="h-5 w-5 mr-2" />
                  Star on GitHub
                </Button>
              </div>
            </motion.div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
