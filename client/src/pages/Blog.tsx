import { motion } from "framer-motion";
import {
  BookOpen,
  Calendar,
  Clock,
  User,
  ArrowRight,
  Code2,
  Package,
  Users,
  GraduationCap,
  Sparkles,
  TrendingUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import Layout from "@/components/Layout";

interface BlogPost {
  id: string;
  title: string;
  excerpt: string;
  date: string;
  author: string;
  authorInitials: string;
  category: string;
  readTime: string;
  featured?: boolean;
}

interface Category {
  name: string;
  icon: React.ElementType;
  count: number;
}

const categories: Category[] = [
  { name: "Engineering", icon: Code2, count: 24 },
  { name: "Product", icon: Package, count: 12 },
  { name: "Community", icon: Users, count: 8 },
  { name: "Tutorials", icon: GraduationCap, count: 18 }
];

const blogPosts: BlogPost[] = [
  {
    id: "1",
    title: "How We Built Our AI-Powered Infrastructure Generator",
    excerpt: "A deep dive into the architecture behind our code analysis engine, including how we use machine learning to understand codebases and generate production-ready infrastructure configurations.",
    date: "November 28, 2025",
    author: "Sarah Mitchell",
    authorInitials: "SM",
    category: "Engineering",
    readTime: "12 min read",
    featured: true
  },
  {
    id: "2",
    title: "Introducing Multi-Region Deployments",
    excerpt: "Deploy your applications to 300+ edge locations worldwide with a single click. Learn how our new multi-region feature works.",
    date: "November 25, 2025",
    author: "Marcus Rodriguez",
    authorInitials: "MR",
    category: "Product",
    readTime: "5 min read"
  },
  {
    id: "3",
    title: "Best Practices for Kubernetes Configurations",
    excerpt: "Learn the patterns and anti-patterns we've discovered from analyzing thousands of Kubernetes deployments.",
    date: "November 22, 2025",
    author: "David Park",
    authorInitials: "DP",
    category: "Tutorials",
    readTime: "8 min read"
  },
  {
    id: "4",
    title: "Building a Developer Community from Scratch",
    excerpt: "How we grew our developer community to 50,000+ members through authentic engagement and valuable content.",
    date: "November 18, 2025",
    author: "Lisa Thompson",
    authorInitials: "LT",
    category: "Community",
    readTime: "6 min read"
  },
  {
    id: "5",
    title: "Zero-Downtime Database Migrations at Scale",
    excerpt: "A technical walkthrough of how we handle database migrations without any service interruption.",
    date: "November 15, 2025",
    author: "Alex Chen",
    authorInitials: "AC",
    category: "Engineering",
    readTime: "10 min read"
  },
  {
    id: "6",
    title: "Getting Started with Infrastructure as Code",
    excerpt: "A beginner-friendly guide to understanding Terraform, Kubernetes, and modern infrastructure patterns.",
    date: "November 12, 2025",
    author: "Emily Watson",
    authorInitials: "EW",
    category: "Tutorials",
    readTime: "7 min read"
  },
  {
    id: "7",
    title: "The Future of Serverless Computing",
    excerpt: "Our vision for the next generation of serverless platforms and how edge computing is changing everything.",
    date: "November 8, 2025",
    author: "Sarah Mitchell",
    authorInitials: "SM",
    category: "Engineering",
    readTime: "9 min read"
  }
];

const featuredPost = blogPosts.find(post => post.featured);
const regularPosts = blogPosts.filter(post => !post.featured);

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

function getCategoryColor(category: string): string {
  switch (category) {
    case "Engineering":
      return "bg-blue-500/10 text-blue-500 border-blue-500/20";
    case "Product":
      return "bg-purple-500/10 text-purple-500 border-purple-500/20";
    case "Community":
      return "bg-green-500/10 text-green-500 border-green-500/20";
    case "Tutorials":
      return "bg-orange-500/10 text-orange-500 border-orange-500/20";
    default:
      return "bg-primary/10 text-primary border-primary/20";
  }
}

export default function Blog() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-blog">
                <BookOpen className="h-3 w-3 mr-1" /> Blog
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-blog-title">
                Engineering <span className="text-primary">Blog</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-blog-subtitle">
                Insights, tutorials, and behind-the-scenes looks at how we're building 
                the future of infrastructure automation.
              </p>
            </motion.div>
          </div>
        </section>

        <section className="py-12">
          <div className="container px-4">
            <div className="grid lg:grid-cols-4 gap-8">
              <div className="lg:col-span-3">
                {featuredPost && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="mb-12"
                  >
                    <div className="flex items-center gap-2 mb-4">
                      <TrendingUp className="h-5 w-5 text-primary" />
                      <h2 className="text-lg font-semibold" data-testid="text-featured-title">Featured Post</h2>
                    </div>
                    <Card
                      className="p-8 bg-card border-border hover-elevate cursor-pointer"
                      data-testid="card-featured-post"
                    >
                      <div className="flex flex-wrap items-center gap-3 mb-4">
                        <Badge variant="outline" className={getCategoryColor(featuredPost.category)} data-testid="badge-featured-category">
                          {featuredPost.category}
                        </Badge>
                        <Badge variant="secondary" data-testid="badge-featured-label">
                          <Sparkles className="h-3 w-3 mr-1" /> Featured
                        </Badge>
                      </div>
                      <h3 className="text-2xl md:text-3xl font-bold mb-4" data-testid="text-featured-post-title">
                        {featuredPost.title}
                      </h3>
                      <p className="text-muted-foreground mb-6 text-lg" data-testid="text-featured-post-excerpt">
                        {featuredPost.excerpt}
                      </p>
                      <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <Avatar className="h-8 w-8">
                            <AvatarFallback className="bg-primary/10 text-primary text-xs">
                              {featuredPost.authorInitials}
                            </AvatarFallback>
                          </Avatar>
                          <span data-testid="text-featured-author">{featuredPost.author}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Calendar className="h-4 w-4" />
                          <span data-testid="text-featured-date">{featuredPost.date}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          <span data-testid="text-featured-readtime">{featuredPost.readTime}</span>
                        </div>
                      </div>
                      <Button className="mt-6" data-testid="button-read-featured">
                        Read Article <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    </Card>
                  </motion.div>
                )}

                <div className="mb-6">
                  <h2 className="text-lg font-semibold" data-testid="text-latest-title">Latest Posts</h2>
                </div>
                <motion.div
                  variants={containerVariants}
                  initial="hidden"
                  animate="visible"
                  className="grid md:grid-cols-2 gap-6"
                >
                  {regularPosts.map((post, index) => (
                    <motion.div key={post.id} variants={itemVariants}>
                      <Card
                        className="p-6 bg-card border-border h-full hover-elevate cursor-pointer flex flex-col"
                        data-testid={`card-blog-post-${post.id}`}
                      >
                        <Badge 
                          variant="outline" 
                          className={`self-start mb-4 ${getCategoryColor(post.category)}`}
                          data-testid={`badge-post-category-${post.id}`}
                        >
                          {post.category}
                        </Badge>
                        <h3 className="font-semibold text-lg mb-2" data-testid={`text-post-title-${post.id}`}>
                          {post.title}
                        </h3>
                        <p className="text-sm text-muted-foreground mb-4 flex-1" data-testid={`text-post-excerpt-${post.id}`}>
                          {post.excerpt}
                        </p>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground mt-auto">
                          <div className="flex items-center gap-2">
                            <Avatar className="h-6 w-6">
                              <AvatarFallback className="bg-primary/10 text-primary text-[10px]">
                                {post.authorInitials}
                              </AvatarFallback>
                            </Avatar>
                            <span data-testid={`text-post-author-${post.id}`}>{post.author}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            <span data-testid={`text-post-date-${post.id}`}>{post.date}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            <span data-testid={`text-post-readtime-${post.id}`}>{post.readTime}</span>
                          </div>
                        </div>
                      </Card>
                    </motion.div>
                  ))}
                </motion.div>

                <div className="mt-10 text-center">
                  <Button variant="outline" size="lg" data-testid="button-load-more">
                    Load More Posts
                  </Button>
                </div>
              </div>

              <div className="lg:col-span-1">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="sticky top-24"
                >
                  <Card className="p-6 bg-card border-border" data-testid="card-categories">
                    <h3 className="font-semibold mb-4" data-testid="text-categories-title">Categories</h3>
                    <div className="space-y-2">
                      {categories.map((category, index) => (
                        <button
                          key={category.name}
                          className="w-full flex items-center justify-between p-3 rounded-md bg-background/50 hover-elevate text-left"
                          data-testid={`button-category-${index}`}
                        >
                          <div className="flex items-center gap-3">
                            <div className="h-8 w-8 rounded-md bg-primary/10 flex items-center justify-center">
                              <category.icon className="h-4 w-4 text-primary" />
                            </div>
                            <span className="text-sm font-medium">{category.name}</span>
                          </div>
                          <Badge variant="secondary" className="text-xs" data-testid={`badge-category-count-${index}`}>
                            {category.count}
                          </Badge>
                        </button>
                      ))}
                    </div>
                  </Card>

                  <Card className="p-6 bg-card border-border mt-6" data-testid="card-newsletter">
                    <h3 className="font-semibold mb-2" data-testid="text-newsletter-title">Subscribe to our newsletter</h3>
                    <p className="text-sm text-muted-foreground mb-4" data-testid="text-newsletter-description">
                      Get the latest posts delivered straight to your inbox.
                    </p>
                    <Button className="w-full" data-testid="button-subscribe">
                      Subscribe
                    </Button>
                  </Card>
                </motion.div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}
