---
sidebar_position: 4
title: User Management
---

# Managing Users & Permissions

This guide explains how to manage users, assign roles, and control access to your Amplifi platform.

## Understanding User Management

Effective user management ensures that:

- The right people have access to the right information
- Sensitive data remains secure
- Administrative tasks are properly assigned
- Users have appropriate permissions for their roles

<!-- Image temporarily removed -->

## User Roles Explained

Amplifi uses a role-based access control system with the following primary roles:

| Role | Access Level | Responsibilities | Typical Position |
|------|-------------|------------------|------------------|
| **Admin** | Complete platform control | Manage users, workspaces, and settings | IT administrator, Department head |
| **Member** | Use existing resources | Access workspaces, search data, create reports | Business analyst, Team member |
| **Developer** | Technical configuration | Set up integrations, configure data sources | Data engineer, Developer |

<!-- Image temporarily removed -->

## Managing Users

### Adding New Users

1. Navigate to **Organization Settings** > **Users**
2. Click **Add User** button
3. Enter the user's:
   - Email address
   - First and last name
   - Role (Admin, Member, or Developer)
   - Department (optional)
4. Click **Send Invitation**

<!-- Image temporarily removed -->

The new user will receive an email invitation to join your Amplifi organization.

### Editing Users

To modify an existing user's details:

1. Go to **Organization Settings** > **Users**
2. Find the user in the list
3. Click the **Edit** icon (pencil)
4. Update their:
   - Name
   - Role
   - Department
   - Contact information
5. Click **Save Changes**

### Deactivating Users

When someone leaves your organization:

1. Navigate to **Organization Settings** > **Users**
2. Find the user in the list
3. Click the **Deactivate** button
4. Confirm the deactivation

Deactivated users:
- Cannot log in to the platform
- Remain in the system for audit purposes
- Can be reactivated later if needed

### Reactivating Users

To restore access for a deactivated user:

1. Go to **Organization Settings** > **Users**
2. Click the **Show Inactive** filter
3. Find the deactivated user
4. Click **Reactivate**
5. Confirm the action

## Workspace Permissions

Beyond organization-level roles, you can set specific permissions for each workspace:

### Assigning Users to Workspaces

1. Navigate to your workspace
2. Click **Settings** > **Team**
3. Select **Add Member**
4. Search for users in your organization
5. Assign one of these workspace roles:
   - **Workspace Admin**: Full control of this workspace
   - **Workspace Member**: Can use the workspace but can't change settings
   - **Workspace Viewer**: Read-only access
6. Click **Add**

<!-- Image temporarily removed -->

### Modifying Workspace Access

To change a user's workspace permissions:

1. Go to the workspace
2. Click **Settings** > **Team**
3. Find the user in the list
4. Use the dropdown menu to change their role
5. Or click **Remove** to revoke access entirely

## Creating Custom Roles (Enterprise Feature)

If the standard roles don't meet your needs, you can create custom roles:

1. Navigate to **Organization Settings** > **Roles**
2. Click **Create Role**
3. Enter a **Role Name** and **Description**
4. Configure permissions for:
   - Data sources (view, edit, create, delete)
   - Destinations (view, edit, create, delete)
   - Models (view, train, deploy)
   - Analytics (view, export)
   - Settings (access, modify)
5. Click **Create**

<!-- Image temporarily removed -->

## Security Best Practices

To maintain platform security:

- **Follow the principle of least privilege** - give users only the access they need
- **Regularly audit user accounts** - remove or downgrade unnecessary access
- **Require strong passwords** - enable password complexity requirements
- **Enable two-factor authentication** - especially for admin accounts
- **Create role-specific training** - ensure users understand their permissions

## User Activity Monitoring

Track how users interact with the platform:

1. Go to **Organization Settings** > **Audit Logs**
2. View activity such as:
   - Logins and login attempts
   - Data source connections
   - Model training activities
   - Search and query history
   - Configuration changes
3. Filter by:
   - User
   - Action type
   - Date range
   - Success/failure status

<!-- Image temporarily removed -->

## Common User Management Scenarios

### Department Reorganization

When teams change:

1. Update user departments in their profiles
2. Reassign workspace access as needed
3. Review role assignments to ensure they match new responsibilities

### Contractor Access

For temporary team members:

1. Add them as regular users with appropriate roles
2. Set a reminder to deactivate their accounts when the contract ends
3. Consider using custom roles with limited permissions

### Mergers or Acquisitions

When combining organizations:

1. Export user lists from both systems
2. Create a plan for role consolidation
3. Add users to the primary Amplifi organization
4. Assign appropriate roles and workspace access

## Troubleshooting Common Issues

### User Can't Access a Workspace

- Verify they have been added to the workspace
- Check their role permissions
- Ensure the workspace is active

### Admin Can't See All Settings

- Verify they have the Admin role at the organization level
- Check if their browser is blocking any content
- Clear browser cache and try again

### Email Invitations Not Received

- Ask the user to check spam/junk folders
- Verify the email address is correct
- Resend the invitation
- Contact your IT department about email filtering

## Next Steps

After setting up your user management:

<!-- 
- [Train your AI models](./training-models) to improve performance
- [Configure data sources](./connecting-data-sources) for your team
- Review [core concepts](../core-concepts/organizations-workspaces) to better understand platform structure -->

Need help with user management? Contact our support team at support@amplifi.io


