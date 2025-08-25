def print_results(step, logger, config):
    if config.verbose:
        string = f"Step {int(step)}: ELBO {float(logger['KL/elbo'][-1]):.4f}; "
        string += f"IW-ELBO {float(logger['logZ/reverse'][-1]):.4f}; "
        if "KL/eubo" in logger and len(logger["KL/eubo"]) > 0:
            string += f"EUBO {float(logger['KL/eubo'][-1]):.4f}; "
        if "discrepancies/sd" in logger and len(logger["discrepancies/sd"]) > 0:
            string += f"SD {float(logger['discrepancies/sd'][-1]):.4f}; "

        try:
            string += f"reverse_ESS {float(logger['ESS/reverse'][-1]):.6f}; "
            if "ESS/forward" in logger and len(logger["ESS/forward"]) > 0:
                string += f"forward_ESS {float(logger['ESS/forward'][-1]):.6f}"
        except:
            pass

        print(string)
