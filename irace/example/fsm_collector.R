args = commandArgs(trailingOnly=TRUE)

readRdata <- function(exp_name) {
        output_file <- file(exp_name)
        content <- c()

        run_file <- paste("./results-folder", "/irace.Rdata", sep="")
        if (file.exists(run_file)) {
                load(run_file)
                pnames <- iraceResults$parameters$names
                switches <- iraceResults$parameters$switches[pnames]
                #Selection of the candidate (best of the last iteration)
                candidate <- iraceResults$allConfigurations[iraceResults$iterationElites[length(iraceResults$iterationElites)] ,pnames]
                fsm <- irace:::buildCommandLine(values=candidate , switches=switches)
                content <- c(content, fsm)
        } else {
                stop("File does not exist!")
        }

        
	writeLines(content, output_file)
        close(output_file)
}




library("irace")
setwd(".")
readRdata("./mission-folder/fsm.txt")


