# PostHog Simple Implementation

## Overview

This demonstrates how to implement PostHog tracking using separate, simple functions for `identify` and `capture` operations. This approach is much clearer and easier to understand.

## What's Been Implemented

### 1. PostHog Utility Functions (`/src/utils/posthogUtils.js`)

Simple, focused functions for PostHog operations:

#### User Identification

```javascript
identifyUser("user123", {
  email: "user@example.com",
  name: "John Doe",
  role: "admin",
});
```

#### Event Capture

```javascript
captureEvent("button_clicked", {
  button_name: "submit",
  location: "form",
  timestamp: new Date().toISOString(),
});
```

#### Combined Operations (when needed)

```javascript
// Step 1: Identify user
identifyUser(user.id, { email: user.email, name: user.name });

// Step 2: Capture event
captureEvent("feature_accessed", { feature_name: "premium_tool" });
```

### 2. Modified DeploymentCard Component

The `DeploymentCard` component now uses the utility functions:

#### Upgrade Button (with User Identification)

```javascript
onClick={() => {
  // Step 1: Identify user first
  if (user) {
    identifyUser(user.id || user.email, {
      email: user.email,
      name: user.name,
      role: user.role,
      client_id: user.clientId
    });
  }

  // Step 2: Capture the button click event
  captureEvent('upgrade_button_clicked', {
    button_type: 'upgrade',
    deployment_title: item?.title,
    deployment_subtitle: item?.subtitle,
    user_identified: !!user
  });
}}
```

#### Read Documentation Button (Event Capture Only)

```javascript
onClick={() => {
  // Just capture the event (no user identification)
  captureEvent('documentation_button_clicked', {
    button_type: 'read_documentation',
    deployment_title: item?.title,
    deployment_subtitle: item?.subtitle,
    user_agent: navigator.userAgent
  });
}}
```

## How to Use

### 1. Import Functions

```javascript
import { identifyUser, captureEvent } from "@/utils/posthogUtils";
```

### 2. User Identification Only

```javascript
// Identify a user with their properties
identifyUser("user123", {
  email: "user@example.com",
  name: "John Doe",
  role: "admin",
});
```

### 3. Event Capture Only

```javascript
// Capture an event without user identification
captureEvent("button_clicked", {
  button_name: "submit",
  location: "form",
  button_color: "blue",
});
```

### 4. Combined Operations (Most Common)

```javascript
// Step 1: Identify the user
identifyUser(user.id, {
  email: user.email,
  name: user.name,
  role: user.role,
});

// Step 2: Capture the event
captureEvent("upgrade_button_clicked", {
  button_type: "upgrade",
  location: "dashboard",
  user_plan: user.plan,
});
```

### 5. Anonymous User Tracking

```javascript
// For users who aren't logged in
captureEvent("anonymous_action", {
  action_type: "page_view",
  referrer: document.referrer,
  user_agent: navigator.userAgent,
});
```

## Events Being Tracked

| Event Name       | When It Fires                  | Properties                                                                       |
| ---------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `button_clicked` | User clicks any tracked button | button_name, timestamp, page_url, user_identified + custom properties            |
| `form_submitted` | User submits any tracked form  | form_type, form_fields, form_data_count, timestamp, page_url + custom properties |
| Custom events    | Any custom event you define    | Full control over event name and properties                                      |

## Testing

1. **Run your app locally**
2. **Click the buttons** in the DeploymentCard component
3. **Check browser console** for confirmation messages
4. **Open Network tab** to see PostHog API requests
5. **Check PostHog dashboard** for the events

## Benefits

- ✅ **Simple & Clear** - Each function has one clear responsibility
- ✅ **Easy to Read** - No complex nested objects or confusing parameters
- ✅ **Two Simple Steps** - First identify, then capture (when needed)
- ✅ **Flexible** - Use identify, capture, or both as needed
- ✅ **Maintainable** - Easy to modify, debug, and understand
- ✅ **Consistent** - Same pattern everywhere in your app
- ✅ **No Additional Setup** - Uses your existing PostHog provider

## Next Steps

- Apply similar tracking to other buttons in your app
- Track form submissions, page navigation, and user interactions
- Create custom events for your specific business logic
- Set up PostHog funnels and dashboards to analyze user behavior
