import { motion } from "framer-motion";
import {
  Briefcase,
  MapPin,
  Clock,
  Building2,
  Heart,
  DollarSign,
  GraduationCap,
  Globe,
  Coffee,
  Laptop,
  Calendar,
  Sparkles,
  ArrowRight,
  Users,
  Rocket,
  Shield
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Layout from "@/components/Layout";
import { Link } from "wouter";

interface Benefit {
  icon: React.ElementType;
  title: string;
  description: string;
}

interface JobPosition {
  id: string;
  title: string;
  department: string;
  location: string;
  type: string;
}

const benefits: Benefit[] = [
  {
    icon: Globe,
    title: "Remote-First",
    description: "Work from anywhere in the world. We have team members across 15+ countries."
  },
  {
    icon: Heart,
    title: "Health & Wellness",
    description: "Comprehensive medical, dental, and vision coverage for you and your family."
  },
  {
    icon: DollarSign,
    title: "Competitive Equity",
    description: "Meaningful ownership stake in the company with employee-friendly terms."
  },
  {
    icon: GraduationCap,
    title: "Learning Budget",
    description: "$2,500 annual budget for courses, conferences, books, and personal development."
  },
  {
    icon: Laptop,
    title: "Equipment Stipend",
    description: "Top-of-the-line equipment and $1,000 home office setup allowance."
  },
  {
    icon: Calendar,
    title: "Flexible PTO",
    description: "Unlimited vacation policy with a minimum of 4 weeks encouraged."
  },
  {
    icon: Coffee,
    title: "Team Retreats",
    description: "Annual company-wide retreats and quarterly team offsites."
  },
  {
    icon: Shield,
    title: "Parental Leave",
    description: "16 weeks paid parental leave for all new parents."
  }
];

const openPositions: JobPosition[] = [
  {
    id: "1",
    title: "Senior Backend Engineer",
    department: "Engineering",
    location: "Remote (US/EU)",
    type: "Full-time"
  },
  {
    id: "2",
    title: "Frontend Engineer",
    department: "Engineering",
    location: "Remote (Worldwide)",
    type: "Full-time"
  },
  {
    id: "3",
    title: "DevOps Engineer",
    department: "Infrastructure",
    location: "Remote (US/EU)",
    type: "Full-time"
  },
  {
    id: "4",
    title: "Developer Advocate",
    department: "Developer Relations",
    location: "Remote (Worldwide)",
    type: "Full-time"
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

function getDepartmentColor(department: string): string {
  switch (department) {
    case "Engineering":
      return "bg-blue-500/10 text-blue-500 border-blue-500/20";
    case "Infrastructure":
      return "bg-purple-500/10 text-purple-500 border-purple-500/20";
    case "Developer Relations":
      return "bg-green-500/10 text-green-500 border-green-500/20";
    default:
      return "bg-primary/10 text-primary border-primary/20";
  }
}

export default function Careers() {
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
              <Badge variant="outline" className="mb-6 text-primary border-primary/30" data-testid="badge-careers">
                <Briefcase className="h-3 w-3 mr-1" /> Careers
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6" data-testid="text-careers-title">
                Work at <span className="text-primary">PlatformBuilder</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto" data-testid="text-careers-subtitle">
                Join a team of passionate engineers, designers, and operators building the future 
                of infrastructure automation. We're remote-first, well-funded, and growing fast.
              </p>

              <div className="flex justify-center gap-8 mt-10 flex-wrap">
                <div className="text-center" data-testid="stat-team-size">
                  <div className="text-4xl font-bold text-primary">45+</div>
                  <div className="text-sm text-muted-foreground">Team Members</div>
                </div>
                <div className="text-center" data-testid="stat-countries">
                  <div className="text-4xl font-bold text-primary">15+</div>
                  <div className="text-sm text-muted-foreground">Countries</div>
                </div>
                <div className="text-center" data-testid="stat-open-roles">
                  <div className="text-4xl font-bold text-primary">{openPositions.length}</div>
                  <div className="text-sm text-muted-foreground">Open Roles</div>
                </div>
              </div>
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
                <Sparkles className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-benefits-title">Why Join Us</h2>
              </div>
              <p className="text-muted-foreground max-w-2xl mx-auto" data-testid="text-benefits-subtitle">
                We believe in taking care of our team so they can do their best work.
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="grid md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
              {benefits.map((benefit, index) => (
                <motion.div key={benefit.title} variants={itemVariants}>
                  <Card
                    className="p-6 bg-card border-border h-full"
                    data-testid={`card-benefit-${index}`}
                  >
                    <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                      <benefit.icon className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="font-semibold text-lg mb-2" data-testid={`text-benefit-title-${index}`}>
                      {benefit.title}
                    </h3>
                    <p className="text-sm text-muted-foreground" data-testid={`text-benefit-description-${index}`}>
                      {benefit.description}
                    </p>
                  </Card>
                </motion.div>
              ))}
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
              className="mb-12"
            >
              <div className="flex items-center gap-2 mb-4">
                <Rocket className="h-5 w-5 text-primary" />
                <h2 className="text-2xl font-bold" data-testid="text-positions-title">Open Positions</h2>
              </div>
              <p className="text-muted-foreground max-w-2xl" data-testid="text-positions-subtitle">
                We're looking for talented individuals to join our growing team. Check out our open roles below.
              </p>
            </motion.div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="space-y-4"
            >
              {openPositions.map((position, index) => (
                <motion.div key={position.id} variants={itemVariants}>
                  <Card
                    className="p-6 bg-card border-border hover-elevate"
                    data-testid={`card-position-${position.id}`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex flex-wrap items-center gap-3 mb-2">
                          <h3 className="font-semibold text-lg" data-testid={`text-position-title-${position.id}`}>
                            {position.title}
                          </h3>
                          <Badge 
                            variant="outline" 
                            className={getDepartmentColor(position.department)}
                            data-testid={`badge-position-department-${position.id}`}
                          >
                            {position.department}
                          </Badge>
                        </div>
                        <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
                          <div className="flex items-center gap-1">
                            <MapPin className="h-4 w-4" />
                            <span data-testid={`text-position-location-${position.id}`}>{position.location}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            <span data-testid={`text-position-type-${position.id}`}>{position.type}</span>
                          </div>
                        </div>
                      </div>
                      <Button data-testid={`button-apply-${position.id}`}>
                        Apply Now <ArrowRight className="ml-2 h-4 w-4" />
                      </Button>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="mt-10"
            >
              <Card className="p-8 bg-background border-border text-center" data-testid="card-general-application">
                <Users className="h-10 w-10 text-primary mx-auto mb-4" />
                <h3 className="font-semibold text-lg mb-2" data-testid="text-general-title">
                  Don't see a role that fits?
                </h3>
                <p className="text-muted-foreground mb-4 max-w-md mx-auto" data-testid="text-general-description">
                  We're always looking for talented individuals. Send us your resume and tell us how you can contribute.
                </p>
                <Button variant="outline" data-testid="button-general-application">
                  Submit General Application
                </Button>
              </Card>
            </motion.div>
          </div>
        </section>

        <section className="py-20 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
          <div className="container px-4 text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-cta-title">
                Ready to Make an Impact?
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8" data-testid="text-cta-subtitle">
                Join us in building tools that developers love. We can't wait to meet you.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="font-bold" data-testid="button-view-all-roles">
                  <Briefcase className="mr-2 h-4 w-4" /> View All Roles
                </Button>
                <Link href="/about">
                  <Button size="lg" variant="outline" data-testid="button-learn-about">
                    Learn About Us <ArrowRight className="ml-2 h-4 w-4" />
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
