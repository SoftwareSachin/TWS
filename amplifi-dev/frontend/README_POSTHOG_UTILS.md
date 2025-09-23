# PostHog Utility Functions - Quick Reference

## Import

```javascript
import {
  trackEvent,
  trackButtonClick,
  trackFormSubmission,
  identifyUser,
  captureEvent,
} from "@/utils/posthogUtils";
```

## Quick Examples

### 1. Button Click Tracking

```javascript
// Simple button click
trackButtonClick("submit_button");

// Button with user identification
trackButtonClick("upgrade", {
  userId: user.id,
  userProps: { email: user.email, name: user.name },
  additionalProps: { button_type: "primary" },
});

// Anonymous button click
trackButtonClick("cta_button", { shouldIdentify: false });
```

### 2. Form Submission Tracking

```javascript
trackFormSubmission("contact_form", {
  userId: user.id,
  userProps: { email: user.email },
  formData: { name: "John", email: "john@example.com" },
});
```

### 3. Custom Event Tracking

```javascript
trackEvent({
  userId: user.id,
  userProps: { email: user.email, role: user.role },
  eventName: "feature_accessed",
  eventProps: { feature_name: "premium_tool" },
});
```

### 4. User Identification Only

```javascript
identifyUser("user123", { email: "user@example.com", name: "John Doe" });
```

### 5. Event Capture Only

```javascript
captureEvent("page_viewed", { page: "dashboard" });
```

## Function Parameters

### trackButtonClick(buttonName, options)

- `buttonName` (string): Name/identifier of the button
- `options.userId` (string): User ID for identification
- `options.userProps` (object): User properties
- `options.additionalProps` (object): Additional event properties
- `options.shouldIdentify` (boolean): Whether to identify user (default: true)

### trackFormSubmission(formType, options)

- `formType` (string): Type of form
- `options.userId` (string): User ID for identification
- `options.userProps` (object): User properties
- `options.formData` (object): Form data for analysis
- `options.additionalProps` (object): Additional event properties

### trackEvent(options)

- `options.userId` (string): User ID for identification
- `options.userProps` (object): User properties
- `options.eventName` (string): Event name to capture
- `options.eventProps` (object): Event properties
- `options.shouldIdentify` (boolean): Whether to identify user (default: true)

## Automatic Properties Added

All events automatically include:

- `timestamp`: ISO string of when event occurred
- `page_url`: Current page URL
- `user_identified`: Boolean indicating if user was identified

## Error Handling

All functions include try-catch blocks and won't break your app if PostHog fails. Errors are logged to console.

## Usage in Components

```javascript
const MyComponent = ({ user }) => {
  const handleClick = () => {
    trackButtonClick("my_button", {
      userId: user?.id,
      userProps: { email: user?.email, role: user?.role },
      additionalProps: { location: "dashboard", button_color: "blue" },
    });

    // Your button logic here
  };

  return <button onClick={handleClick}>Click Me</button>;
};
```
