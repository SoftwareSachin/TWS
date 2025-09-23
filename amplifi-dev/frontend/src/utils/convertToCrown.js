/**
 * The function generates a cron expression based on the trigger time and frequency provided.
 * @param triggerTime - The `triggerTime` parameter is an object that contains a `value` property
 * representing the time of day when the cron job should be triggered. The possible values for
 * `triggerTime.value` are "morning", "afternoon", or "evening".
 * @param frequency - The `frequency` parameter in the `generateCronExpression` function determines how
 * often the cron job should run. It can have three possible values: "daily", "weekly", or "monthly".
 * @returns The `generateCronExpression` function returns a cron expression based on the `triggerTime`
 * and `frequency` parameters provided. The cron expression generated depends on the values of
 * `triggerTime` and `frequency`:
 */
export const generateCronExpression = (triggerTime, frequency) => {
  const timeMapping = {
    morning: "8",
    afternoon: "12",
    evening: "18",
  };

  const time = timeMapping[triggerTime?.value] || "*";

  switch (frequency?.value) {
    case "daily":
      return `0 ${time} * * *`; // Every day at the specified time
    case "weekly":
      return `0 ${time} * * 0`; // Every week at the specified time (Sunday)
    case "monthly":
      return `0 ${time} 1 * *`; // Every month on the 1st at the specified time
    default:
      return "";
  }
};
