/**
 * Reverse function to parse a cron expression and set Trigger Time and Frequency
 */
export const reverseCronExpression = (cronExpression) => {
  console.log("testing --- >", cronExpression);

  if (!cronExpression || typeof cronExpression !== "string") {
    return { triggerTime, frequency };
  }

  // Split the cron expression into its parts
  const [minute, hour, dayOfMonth, month, dayOfWeek] =
    cronExpression.split(" ");

  // Reverse map frequency
  let frequency = null;

  if (dayOfMonth === "*" && month === "*" && dayOfWeek === "*") {
    frequency = { value: "daily", label: "Daily" };
  } else if (dayOfMonth === "*" && month === "*" && dayOfWeek !== "*") {
    frequency = { value: "weekly", label: "Weekly" };
  } else if (dayOfMonth === "1" && month === "*" && dayOfWeek === "*") {
    frequency = { value: "monthly", label: "Monthly" };
  } else if (dayOfMonth === "1" && month === "1" && dayOfWeek === "*") {
    frequency = { value: "yearly", label: "Yearly" };
  } else if (
    minute === "*" &&
    hour === "*" &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    frequency = { value: "hourly", label: "Hourly" };
  }

  // Reverse map trigger time
  let triggerTime = null;

  // Parse the hour part, considering ranges and lists
  const hourInt = parseInt(hour, 10);

  if (!isNaN(hourInt)) {
    if (hourInt >= 6 && hourInt < 12) {
      triggerTime = { value: "morning", label: "Morning" };
    } else if (hourInt >= 12 && hourInt < 18) {
      triggerTime = { value: "afternoon", label: "Afternoon" };
    } else if (hourInt >= 18 && hourInt < 24) {
      triggerTime = { value: "evening", label: "Evening" };
    } else if (hourInt >= 0 && hourInt < 6) {
      triggerTime = { value: "night", label: "Night" };
    }
  } else if (hour.includes(",")) {
    // Handle lists like "6,12,18"
    triggerTime = { value: "multiple", label: "Multiple Times" };
  } else if (hour.includes("*/")) {
    // Handle intervals like "*/2"
    triggerTime = { value: "interval", label: "Interval" };
  }

  return { triggerTime, frequency };
};
