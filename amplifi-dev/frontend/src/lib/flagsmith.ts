// lib/flagsmith.ts
import flagsmith from "flagsmith";

export const initFlagsmith = async () => {
  const envKey = process.env.NEXT_PUBLIC_FLAGSMITH_ENVIRONMENT_KEY;
  if (!envKey) {
    return null; // or return a no-op object if your code expects something
  }

  await flagsmith.init({
    environmentID: envKey,
    cacheFlags: true,
  });

  return flagsmith;
};
