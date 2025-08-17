def print_results(step, logger, config):
    if config.verbose:
        string = f"Step {int(step)}: ELBO {float(logger['KL/elbo'][-1]):.4f}; "
        string += f"IW-ELBO {float(logger['logZ/reverse'][-1]):.4f}; "
        if "KL/eubo" in logger:
            string += f"EUBO {float(logger['KL/eubo'][-1]):.4f}; "

        string += f"reverse_ESS {float(logger['ESS/reverse'][-1]):.6f}; "
        if "ESS/forward" in logger:
            string += f"forward_ESS {float(logger['ESS/forward'][-1]):.6f}"

        print(string)
