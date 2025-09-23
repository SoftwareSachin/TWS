# Simple PostHog Guide - Using Separate Functions

## Overview

This guide shows the **simple and clear approach** to PostHog tracking using separate functions for `identify` and `capture` operations.

## The Two Functions You Need

### 1. `identifyUser(userId, userProps)`

**Purpose:** Identifies a user with their properties

```javascript
identifyUser("user123", {
  email: "john@example.com",
  name: "John Doe",
  role: "admin",
});
```

### 2. `captureEvent(eventName, eventProps)`

**Purpose:** Captures an event with custom properties

```javascript
captureEvent("button_clicked", {
  button_name: "submit",
  location: "form",
});
```

## How to Use in Your Buttons

### For Authenticated Users

```javascript
const handleClick = () => {
  // Step 1: Identify the user
  identifyUser(user.id, {
    email: user.email,
    name: user.name,
    role: user.role,
  });

  // Step 2: Capture the event
  captureEvent("button_clicked", {
    button_name: "upgrade",
    location: "dashboard",
  });

  // Your button logic here
};
```

### For Anonymous Users

```javascript
const handleClick = () => {
  // Just capture the event (no user identification)
  captureEvent("button_clicked", {
    button_name: "cta",
    location: "landing_page",
  });

  // Your button logic here
};
```

## Real Examples from Your Code

### Upgrade Button (with user identification)

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

### Read Documentation Button (no user identification)

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

## Why This Approach is Better

✅ **Simple** - Each function does one thing  
✅ **Clear** - Easy to understand what's happening  
✅ **Flexible** - Use identify, capture, or both  
✅ **Maintainable** - Easy to modify and debug  
✅ **Consistent** - Same pattern everywhere

## Import Statement

```javascript
import { identifyUser, captureEvent } from "@/utils/posthogUtils";
```

## Common Use Cases

### Button Clicks

```javascript
captureEvent("button_clicked", { button_name: "submit" });
```

### Form Submissions

```javascript
identifyUser(user.id, { email: user.email });
captureEvent("form_submitted", { form_type: "contact" });
```

### Page Views

```javascript
captureEvent("page_viewed", { page: "dashboard" });
```

### Feature Access

```javascript
identifyUser(user.id, { email: user.email, plan: user.plan });
captureEvent("feature_accessed", { feature: "premium_tool" });
```

## That's It!

No complex objects, no confusing parameters, just two simple functions that do exactly what they say they do.
