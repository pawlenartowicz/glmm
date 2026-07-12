#!/usr/bin/env Rscript
# Static HTML report over the Study-A analysis (status_map.csv from
# analyze_grid.R). Structures grouped 4 per table in n_theta order: rows =
# size x balance/regime, columns = structure x family (two-row header); each
# cell shows glmm vs MixedModels wall time + eval count, colored by the
# wall-time ratio. Usage:
#   Rscript grid_report.R [status_map.csv] [out.html]
# Timing caveat rendered into the page header: walls are meaningful only for
# clock-locked passes (run_meta no_turbo==1). MixedModels walls are now hot --
# grid_fit.jl warm-up-fits each cell before the timed fit (compile_seconds
# recorded as proof), so the JIT-inflation caveat no longer applies.
args <- commandArgs(TRUE)
parity_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
csv <- if (length(args) >= 1) args[1] else
  file.path(parity_dir, "reports", "grid", "status_map.csv")
out <- if (length(args) >= 2) args[2] else
  file.path(dirname(csv), "report.html")
res <- read.csv(csv, stringsAsFactors = FALSE)
res$size <- sub(".*_(g[0-9]+p[0-9]+)_.*", "\\1", res$case_id)
res$g <- as.integer(sub("g([0-9]+)p.*", "\\1", res$size))
res$p <- as.integer(sub(".*p([0-9]+)$", "\\1", res$size))
res$variant <- paste(res$balance, res$regime, sep = "/")
res$wall_ratio <- ifelse(res$glmm_status == "ok" & res$mm_status == "ok",
                         res$wall_glmm / res$wall_mm, NA_real_)

fmt_t <- function(s) ifelse(s < 0.9995, sprintf("%.1f&thinsp;ms", s * 1000),
                            sprintf("%.2f&thinsp;s", s))
fmt_r <- function(r) ifelse(r < 0.095, sprintf("&times;%.3f", r),
                            sprintf("&times;%.2f", r))

# color: log2(wall glmm/MM) -> green (<=0.5, glmm 2x faster) .. yellow (1) ..
# orange (>=2, glmm 2x slower); pastel hsl backgrounds, clamped at ratio 4x
cell_color <- function(r) {
  lr <- max(-2, min(2, log2(r)))
  hue <- if (lr <= 0) 60 - lr * 30 else 60 - lr * 15
  sprintf("hsl(%.0f,75%%,85%%)", hue)
}

cell_html <- function(row) {
  if (row$glmm_status != "ok" && row$mm_status != "ok")
    return(sprintf('<td class="both-fail" title="%s">glmm: %s<br>MM.jl: %s</td>',
                   row$case_id, row$glmm_status, row$mm_status))
  if (row$glmm_status != "ok")
    return(sprintf('<td class="glmm-fail" title="%s">glmm: %s<br>MM.jl %s</td>',
                   row$case_id, row$glmm_status, fmt_t(row$wall_mm)))
  if (row$mm_status != "ok")
    return(sprintf('<td class="mm-fail" title="%s">glmm %s &middot; %dev<br>MM.jl: %s</td>',
                   row$case_id, fmt_t(row$wall_glmm), row$n_eval_glmm, row$mm_status))
  mism <- if (row$status == "mismatch") ' mismatch' else ''
  warn <- if (row$status == "mismatch") ' &#9888;' else ''
  sprintf(paste0(
    '<td class="ok%s" style="background:%s" title="%s\nwall ratio %s / eval ratio %s">',
    '<b>%s</b>%s <span class="ev">%d&thinsp;ev</span><br>%s <span class="ev">%d&thinsp;ev</span></td>'),
    mism, cell_color(row$wall_ratio), row$case_id,
    fmt_r(row$wall_ratio), fmt_r(row$eval_ratio),
    fmt_t(row$wall_glmm), warn, row$n_eval_glmm,
    fmt_t(row$wall_mm), row$n_eval_mm)
}

esc <- function(x) gsub("<", "&lt;", x)
html <- c(sprintf(paste0(
  '<meta charset="utf-8"><title>GLMM grid report</title><style>',
  'body{font-family:system-ui,sans-serif;margin:1.5em;font-size:14px}',
  'table{border-collapse:collapse;margin:0 0 2em}',
  'td,th{border:1px solid #bbb;padding:3px 8px;text-align:left;white-space:nowrap}',
  'th{background:#f0f0f0} td b{font-weight:600}',
  '.ev{color:#666;font-size:11px}',
  '.glmm-fail{background:hsl(0,75%%,80%%)} .mm-fail{background:hsl(215,75%%,85%%)}',
  '.both-fail{background:#ddd;color:#666}',
  '.mismatch{outline:2px dashed #c00;outline-offset:-2px}',
  '.grp{border-left:3px solid #888}',
  '.legend span{padding:2px 8px;margin-right:6px;border:1px solid #bbb}',
  'h2{margin:1.2em 0 .3em} .note{color:#555;max-width:70em}</style>',
  '<h1>Optimizer-grid Study A &mdash; glmm vs MixedModels.jl (run 2026-07-11)</h1>',
  '<p class="note">Cell: top line glmm, bottom line MixedModels &mdash; per-fit wall time and optimizer eval count. ',
  'Color = wall-time ratio glmm/MM (hover for exact wall + eval ratios). ',
  'Timing is the aim; eval ratio is the optimizer-side diagnostic. ',
  '</p><p class="note"><b>Environment:</b> Intel Core Ultra 7 265H (Arrow Lake-H), 32&thinsp;GB RAM; ',
  'both engines pinned to P-core 1 (<code>taskset -c 1</code>); clock locked for the whole run: ',
  'governor <code>performance</code>, turbo off (<code>no_turbo=1</code>), P-cores fixed at the 2.2&thinsp;GHz base. ',
  'Timings are only ever produced on this locked configuration &mdash; unlocked passes are excluded by protocol. ',
  'Julia 1.12.6 / MixedModels.jl per the pinned parity env.</p>',
  '<p class="legend"><span style="background:hsl(120,75%%,85%%)">glmm &ge;2&times; faster</span>',
  '<span style="background:hsl(60,75%%,85%%)">even</span>',
  '<span style="background:hsl(30,75%%,85%%)">glmm &ge;2&times; slower</span>',
  '<span style="background:hsl(0,75%%,80%%)">glmm failed</span>',
  '<span style="background:hsl(215,75%%,85%%)">only glmm converged</span>',
  '<span style="background:#ddd">both failed</span>',
  '<span style="outline:2px dashed #c00">&#9888; &beta; mismatch</span></p>',
  '<p><b>Overall</b> (%d jointly-ok of %d cells): wall ratio median %.3f / p90 %.2f; ',
  'eval ratio median %.3f / p90 %.2f.</p>'),
  sum(!is.na(res$wall_ratio)), nrow(res),
  median(res$wall_ratio, na.rm = TRUE), quantile(res$wall_ratio, .9, na.rm = TRUE),
  median(res$eval_ratio, na.rm = TRUE), quantile(res$eval_ratio, .9, na.rm = TRUE)))

structures <- unique(res$structure[order(res$n_theta, res$structure)])
for (grp in split(structures, ceiling(seq_along(structures) / 4))) {
  s <- res[res$structure %in% grp, ]
  # two-row header: structure (n_theta) spanning its families, families below
  hdr1 <- '<tr><th rowspan="2">cell</th>'
  hdr2 <- '<tr>'
  cols <- list()  # (structure, family) pairs in column order
  for (st in grp) {
    fams <- sort(unique(s$family[s$structure == st]))
    hdr1 <- c(hdr1, sprintf('<th colspan="%d" class="grp">%s <small>(n_theta %d)</small></th>',
                            length(fams), esc(st),
                            unique(s$n_theta[s$structure == st])))
    hdr2 <- c(hdr2, sprintf('<th%s>%s</th>',
                            c(' class="grp"', rep('', length(fams) - 1)), fams))
    cols <- c(cols, lapply(fams, function(f) c(st, f)))
  }
  html <- c(html, '<table>', hdr1, '</tr>', hdr2, '</tr>')
  keys <- unique(s[order(s$g, s$p, s$variant), c("size", "variant")])
  for (i in seq_len(nrow(keys))) {
    row_cells <- vapply(seq_along(cols), function(j) {
      r <- s[s$size == keys$size[i] & s$variant == keys$variant[i] &
             s$structure == cols[[j]][1] & s$family == cols[[j]][2], ]
      if (nrow(r) == 0) {
        if (j > 1 && !identical(cols[[j]][1], cols[[j - 1]][1]))
          '<td class="grp"></td>' else '<td></td>'
      } else {
        cell <- cell_html(r[1, ])
        if (j > 1 && !identical(cols[[j]][1], cols[[j - 1]][1]))
          sub('<td class="', '<td class="grp ', cell, fixed = TRUE) else cell
      }
    }, "")
    html <- c(html, sprintf('<tr><th>%s %s</th>', keys$size[i], keys$variant[i]),
              row_cells, '</tr>')
  }
  html <- c(html, '</table>')
}
writeLines(html, out)
cat("report:", out, "\n")
