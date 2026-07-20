# Unified population generator — replaces pop_generate.sh (900 R-process spawns)
# with a single R session. Replicates generate.R exactly: for every (seed, K, d,
# omega) it calls set.seed(seed) immediately before one MixSim() draw, so output
# is byte-identical to `Rscript generate.R <seed> <K> <d> <omega>`.
#
# Resumable: skips a config whose three output files already exist.
# Run from simulation/generated_pop/ :
#   Rscript generate_all.R
#
# Requires the MixSim package. If it is installed in a user library that is not
# on the default .libPaths(), point R_MIXSIM_LIB at that directory, e.g.
#   R_MIXSIM_LIB=~/R/win-library/4.6 Rscript generate_all.R

userlib <- Sys.getenv("R_MIXSIM_LIB", unset = NA)
if (!is.na(userlib) && nzchar(userlib)) .libPaths(c(userlib, .libPaths()))
suppressMessages(library(MixSim))

outdir <- "true_param"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

seeds <- 1:300

# (K, d, omega) configs to generate.
# Main grid (Figs 2-6): K=5, d=10, omega in {0.1, 0.2, 0.3}
# Fig-7 extras (distance concentration): (K,d) in {(5,2),(10,10)} at omega 0.3
configs <- list()
for (om in c(0.1, 0.2, 0.3)) configs[[length(configs) + 1]] <- list(K = 5,  d = 10, omega = om)
configs[[length(configs) + 1]] <- list(K = 5,  d = 2,  omega = 0.3)
configs[[length(configs) + 1]] <- list(K = 10, d = 10, omega = 0.3)

fpath <- function(kind, seed, K, d, omega) {
  file.path(outdir, paste0(kind, "_seed_", as.character(seed),
                           "_ncomp_", as.character(K),
                           "_d_", as.character(d),
                           "_maxoverlap_", as.character(omega), ".txt"))
}

n_done <- 0L; n_skip <- 0L; n_fail <- 0L
t0 <- Sys.time()
for (cfg in configs) {
  K <- cfg$K; d <- cfg$d; omega <- cfg$omega
  for (seed in seeds) {
    wf <- fpath("weights", seed, K, d, omega)
    mf <- fpath("means",   seed, K, d, omega)
    cf <- fpath("covs",    seed, K, d, omega)
    if (file.exists(wf) && file.exists(mf) && file.exists(cf)) { n_skip <- n_skip + 1L; next }

    set.seed(seed)
    param <- tryCatch(
      MixSim(MaxOmega = omega, K = K, p = d, sph = FALSE, hom = FALSE),
      error = function(e) NULL)
    if (is.null(param) || isTRUE(param$fail != 0)) {
      n_fail <- n_fail + 1L
      cat(sprintf("FAIL K=%d d=%d omega=%s seed=%d\n", K, d, as.character(omega), seed))
      next
    }
    write.table(param$Pi, file = wf, row.names = FALSE, col.names = FALSE)
    write.table(param$Mu, file = mf, row.names = FALSE, col.names = FALSE)
    write.table(param$S,  file = cf, row.names = FALSE, col.names = FALSE)
    n_done <- n_done + 1L
  }
  cat(sprintf("done config K=%d d=%d omega=%s | wrote=%d skip=%d fail=%d | %.1fs\n",
              K, d, as.character(omega), n_done, n_skip, n_fail,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
cat(sprintf("ALL DONE wrote=%d skip=%d fail=%d\n", n_done, n_skip, n_fail))
