# Comparison figure: glmm vs the published R GLMM packages, parameter-recovery
# RMSE on the Li & Signorelli (2026) design. Small multiples (one panel per
# parameter, free scales), bars sorted best->worst within each panel; glmm's two
# arms highlighted, the eight published engines as a neutral field.
suppressMessages({ library(ggplot2); library(scales) })
ACC <- dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(ACC) || !nzchar(ACC)) ACC <- getwd()
rep <- read.csv(file.path(ACC, "results", "accuracy_report.csv"))

P <- c("beta0","beta1","beta2","beta3","tau0","tau1","rho01")
plabel <- c(beta0="beta[0]", beta1="beta[1]", beta2="beta[2]", beta3="beta[3]",
            tau0="tau[0]", tau1="tau[1]", rho01="rho['01']")
keep <- c("lme4_LA","lme4_AGQ","GLMMadaptive","glmmTMB","MASS","hglm","brms",
          "rstanarm","glmm_LA","glmm_AGQ")

d <- subset(rep, metric == "rmse" & package %in% keep)
long <- do.call(rbind, lapply(P, function(p)
  data.frame(package = d$package, param = p, value = d[[p]])))
long <- long[is.finite(long$value), ]
long$param <- factor(long$param, levels = P, labels = plabel[P])
# order bars within each panel (best/lowest RMSE at top)
long$row <- factor(paste(long$param, long$package),
                   levels = paste(long$param, long$package)[order(long$param, -long$value)])
long$grp <- ifelse(long$package == "glmm_LA", "glmm (Laplace)",
             ifelse(long$package == "glmm_AGQ", "glmm (AGQ)", "other packages"))
long$grp <- factor(long$grp, levels = c("glmm (AGQ)","glmm (Laplace)","other packages"))

pal <- c("glmm (AGQ)" = "#e8820c", "glmm (Laplace)" = "#2f6fd0", "other packages" = "#c2c7cc")

p <- ggplot(long, aes(row, value, fill = grp)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%.2f", value)), hjust = -0.15, size = 2.7,
            colour = "#33383d") +
  facet_wrap(~param, scales = "free", ncol = 4, labeller = label_parsed) +
  scale_x_discrete(labels = function(x) sub("^.* ", "", x)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
  scale_fill_manual(values = pal, name = NULL) +
  coord_flip() +
  labs(
    title = "glmm vs. R packages for GLMMs — parameter-recovery RMSE",
    subtitle = "Lower is better. Li & Signorelli (2026) design, S=1000; published columns are their frozen Tables 4/5.\nglmm at 1000 reps (single-thread, single-core); AGQ arm on the 12 random-intercept cells only.",
    caption = "Bars sorted best→worst within each panel. Fixed effects (top row) β; variance components (bottom) τ, ρ.",
    x = NULL, y = "mean RMSE (vs. true parameter)") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(colour = "#ececec"),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(colour = "#5b6167", size = 8.5, lineheight = 1.1),
    plot.caption = element_text(colour = "#8a9096", size = 7.5),
    axis.text.y = element_text(size = 8.5))

ggsave(file.path(ACC, "reports", "glmm_vs_engines_rmse.png"), p,
       width = 12, height = 6.3, dpi = 150, bg = "white")
cat("wrote reports/glmm_vs_engines_rmse.png\n")
