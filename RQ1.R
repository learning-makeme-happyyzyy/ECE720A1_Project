# =============================================================================
# Replication of RQ1 from Nagy & Abdalkareem (2022) MSR 2022
# "On the Co-Occurrence of Refactoring of Test and Source Code"
# 
# This script reproduces the descriptive statistics and visualizations for RQ1
# =============================================================================

# Load required libraries
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(scales)

# Set working directory (modify this path to where your CSV files are located)
# setwd("path/to/your/data")

# =============================================================================
# Data Loading and Preparation
# =============================================================================

# Function to extract and process RefactoringMiner data
extract_rminer_data <- function() {
  if (!file.exists('rMinerRefactorings.csv')) {
    stop("File 'rMinerRefactorings.csv' not found. Please ensure it's in the working directory.")
  }
  
  rminer_data <- read_csv('rMinerRefactorings.csv', col_types = cols(
    commit_id = col_character(),
    type = col_character(),
    test = col_character()
  ))
  
  return(rminer_data)
}

# Function to extract and process refactoring commits data
extract_r_commits <- function() {
  if (!file.exists('refactoring_commits.csv')) {
    stop("File 'refactoring_commits.csv' not found. Please ensure it's in the working directory.")
  }
  
  refactoring_commits <- read_csv('refactoring_commits.csv', col_types = cols(
    url = col_character(),
    commit_id = col_character()
  ))
  
  return(refactoring_commits)
}

# =============================================================================
# Commit Classification
# =============================================================================

# Function to classify commits based on refactoring types
classify_commits <- function(rminer_data, refactoring_commits) {
  # Classify each refactoring operation first
  classified_refactorings <- rminer_data %>%
    mutate(
      refactoring_category = case_when(
        test == "True" ~ "test",
        test == "False" ~ "production",
        TRUE ~ "unknown"
      )
    )
  
  # Classify each commit based on its refactorings
  commit_classification <- classified_refactorings %>%
    group_by(commit_id) %>%
    summarise(
      has_production = any(refactoring_category == "production"),
      has_test = any(refactoring_category == "test"),
      .groups = 'drop'
    ) %>%
    mutate(
      classification = case_when(
        has_production & has_test ~ "co-occur",
        has_production & !has_test ~ "production-only",
        !has_production & has_test ~ "test-only",
        TRUE ~ "unknown"
      )
    ) %>%
    select(commit_id, classification)
  
  # Join with repository information
  repo_commit_classification <- refactoring_commits %>%
    left_join(commit_classification, by = "commit_id") %>%
    filter(!is.na(classification) & classification != "unknown")
  
  return(repo_commit_classification)
}

# =============================================================================
# Statistical Analysis and Visualization
# =============================================================================

# Function to calculate repository statistics and create visualization
analyze_and_plot_commit_distribution <- function(repo_commit_classification) {
  # Calculate repository-level statistics
  repo_stats <- repo_commit_classification %>%
    group_by(url, classification) %>%
    summarise(count = n(), .groups = 'drop_last') %>%
    mutate(
      total_refactorings = sum(count),
      percentage = (count / total_refactorings) * 100
    ) %>%
    ungroup()
  
  # Create a wide format for easier analysis
  repo_stats_wide <- repo_stats %>%
    select(url, classification, percentage) %>%
    pivot_wider(
      names_from = classification,
      values_from = percentage,
      values_fill = 0
    ) %>%
    left_join(
      repo_commit_classification %>%
        group_by(url) %>%
        summarise(total_refactorings = n()),
      by = "url"
    )
  
  # Print summary statistics
  cat("Summary Statistics for Commit Classification:\n")
  cat("=============================================\n")
  
  categories <- c("co-occur", "production-only", "test-only")
  for (category in categories) {
    if (category %in% colnames(repo_stats_wide)) {
      cat(sprintf("\n%s:\n", category))
      cat(sprintf("  Mean: %.2f%%\n", mean(repo_stats_wide[[category]], na.rm = TRUE)))
      cat(sprintf("  Median: %.2f%%\n", median(repo_stats_wide[[category]], na.rm = TRUE)))
      cat(sprintf("  Min: %.2f%%\n", min(repo_stats_wide[[category]], na.rm = TRUE)))
      cat(sprintf("  Max: %.2f%%\n", max(repo_stats_wide[[category]], na.rm = TRUE)))
    }
  }
  
  # Prepare data for boxplot
  boxplot_data <- repo_stats_wide %>%
    select(url, `co-occur`, `production-only`, `test-only`) %>%
    pivot_longer(
      cols = c(`co-occur`, `production-only`, `test-only`),
      names_to = "commit_type",
      values_to = "percentage"
    ) %>%
    mutate(
      commit_type = factor(
        commit_type,
        levels = c("production-only", "test-only", "co-occur"),
        labels = c("Source", "Test", "Co-Occurring")
      )
    )
  
  # Create boxplot (equivalent to Figure 1 in the paper)
  p1 <- ggplot(boxplot_data, aes(x = percentage, y = commit_type)) +
    geom_boxplot(fill = "#A0AECA", color = "black", alpha = 0.7) +
    labs(
      title = "Distribution of Refactoring Commit Types",
      x = "Percentage of Commits",
      y = "Refactoring Commits"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank()
    )
  
  print(p1)
  ggsave("PercentageOfCommitsType.pdf", p1, width = 6, height = 4)
  
  return(repo_stats_wide)
}

# =============================================================================
# Test Refactoring Types Analysis (Co-occurring commits)
# =============================================================================

# Function to analyze test refactoring types in co-occurring commits
analyze_test_refactoring_types <- function(rminer_data, repo_commit_classification) {
  # Identify co-occurring commits
  co_occurring_commits <- repo_commit_classification %>%
    filter(classification == "co-occur") %>%
    pull(commit_id)
  
  # Extract test refactorings from co-occurring commits
  test_refactorings_co_occur <- rminer_data %>%
    filter(commit_id %in% co_occurring_commits & test == "True") %>%
    count(type, name = "count") %>%
    arrange(desc(count))
  
  # Print top test refactoring types
  cat("\n\nTop Test Refactoring Types in Co-occurring Commits:\n")
  cat("===================================================\n")
  print(head(test_refactorings_co_occur, 10))
  
  # Get per-project counts for test refactorings in co-occurring commits
  repo_test_refactorings <- rminer_data %>%
    filter(commit_id %in% co_occurring_commits & test == "True") %>%
    left_join(
      repo_commit_classification %>% select(commit_id, url),
      by = "commit_id"
    ) %>%
    count(url, type, name = "frequency")
  
  # Get top 10 most frequent test refactoring types
  top_10_types <- test_refactorings_co_occur %>%
    head(10) %>%
    pull(type)
  
  # Prepare data for boxplot (top 10 types)
  # First, ensure we have all combinations of url and top_10_types
  all_combinations <- expand.grid(
    url = unique(repo_test_refactorings$url),
    refactoring_type = top_10_types,
    stringsAsFactors = FALSE
  )
  
  boxplot_data_top10 <- all_combinations %>%
    left_join(repo_test_refactorings, by = c("url", "refactoring_type" = "type")) %>%
    mutate(
      frequency = ifelse(is.na(frequency), 0, frequency),
      refactoring_type = factor(refactoring_type, levels = rev(top_10_types))
    )
  
  # Create boxplot for top 10 test refactoring types (equivalent to Figure 2 in the paper)
  p2 <- ggplot(boxplot_data_top10, aes(x = frequency, y = refactoring_type)) +
    geom_boxplot(fill = "#A0AECA", color = "black", alpha = 0.7) +
    labs(
      title = "Top 10 Test Refactoring Types in Co-occurring Commits",
      x = "Frequency",
      y = "Refactoring Type"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank()
    )
  
  print(p2)
  ggsave("CoOccurringTestTypesTop10.pdf", p2, width = 8, height = 6)
  
  return(list(
    all_test_refactorings = test_refactorings_co_occur,
    top_10_test_refactorings = head(test_refactorings_co_occur, 10)
  ))
}

# =============================================================================
# Main Execution
# =============================================================================

main <- function() {
  cat("Starting RQ1 Replication Analysis...\n")
  cat("====================================\n\n")
  
  tryCatch({
    # Step 1: Load and prepare data
    cat("Step 1: Loading data from CSV files...\n")
    rminer_data <- extract_rminer_data()
    refactoring_commits <- extract_r_commits()
    
    cat(sprintf("  - Loaded %d refactoring operations\n", nrow(rminer_data)))
    cat(sprintf("  - Loaded %d repository-commit mappings\n", nrow(refactoring_commits)))
    
    # Step 2: Classify commits
    cat("\nStep 2: Classifying commits...\n")
    repo_commit_classification <- classify_commits(rminer_data, refactoring_commits)
    
    cat(sprintf("  - Classified %d commit-repository pairs\n", nrow(repo_commit_classification)))
    
    # Step 3: Analyze commit distribution and create visualization
    cat("\nStep 3: Analyzing commit distribution...\n")
    repo_stats <- analyze_and_plot_commit_distribution(repo_commit_classification)
    
    # Step 4: Analyze test refactoring types in co-occurring commits
    cat("\nStep 4: Analyzing test refactoring types in co-occurring commits...\n")
    test_refactoring_analysis <- analyze_test_refactoring_types(
      rminer_data, 
      repo_commit_classification
    )
    
    cat("\n\nAnalysis completed successfully!\n")
    cat("Generated files:\n")
    cat("  - PercentageOfCommitsType.pdf (Figure 1 equivalent)\n")
    cat("  - CoOccurringTestTypesTop10.pdf (Figure 2 equivalent)\n")
    
    # Return results for further inspection if needed
    return(list(
      repo_stats = repo_stats,
      test_refactoring_analysis = test_refactoring_analysis
    ))
    
  }, error = function(e) {
    cat(sprintf("Error during analysis: %s\n", e$message))
    cat(sprintf("Error traceback: %s\n", e$call))
    return(NULL)
  })
}

# Execute the main function
results <- main()

# Print final summary if analysis was successful
if (!is.null(results)) {
  cat("\n\nFinal Summary:\n")
  cat("=============\n")
  
  # Calculate overall percentages across all repositories
  overall_stats <- results$repo_stats %>%
    summarise(
      across(
        c(`co-occur`, `production-only`, `test-only`),
        list(mean = ~mean(., na.rm = TRUE), median = ~median(., na.rm = TRUE))
      )
    )
  
  cat("Overall percentages across all repositories:\n")
  cat(sprintf("  Source-only: %.1f%% (mean), %.1f%% (median)\n", 
              overall_stats$`production-only_mean`, overall_stats$`production-only_median`))
  cat(sprintf("  Test-only: %.1f%% (mean), %.1f%% (median)\n", 
              overall_stats$`test-only_mean`, overall_stats$`test-only_median`))
  cat(sprintf("  Co-occurring: %.1f%% (mean), %.1f%% (median)\n", 
              overall_stats$`co-occur_mean`, overall_stats$`co-occur_median`))
  
  # Print the top test refactoring types
  cat("\nTop 10 Test Refactoring Types in Co-occurring Commits:\n")
  print(results$test_refactoring_analysis$top_10_test_refactorings)
}