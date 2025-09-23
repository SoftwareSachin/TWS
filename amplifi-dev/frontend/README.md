#project name - Amplifi

This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Rendering Mode](#rendering-mode)
- [Development](#development)
- [Deployment](#deployment)
- [Learn More](#learn-more)

---

## Project Overview

[This is the amplifi frontend application .]

- **Framework**: Next.js
- **Rendering**:
  - Root-level files: Server-Side Rendering (SSR) - layout.js of every component
  - All other files: Client-Side Rendering (CSR) - Files having "use client" on the top
- **API Calls**: Currently, no server-side API calls .
- **Features**: [
  - login with login radius
  - Create user by sending link
  - Auto login for the first time user and reset password
  - workspace listing and creation
  - File upload and Azure Blob
  - File Details
  - Dataset create and file ingest and check status
  - Dataset chunk details
  - Search on dataset
  - Destination listing and creating
  - workflow creation , details , edit , start , stop
  - logout
  - refresh token  
    ]

---

## Getting Started

Follow the steps below to get the application running on your local machine.

- clone the project from repo go inside the frontend folder - cd frontend
- to install packages - do - npm i
- add .env file to the root file . root file where the package.json present .
- add these variables and your values to the file -
  <!-- Add backend base url here  -->
  NEXT_PUBLIC_BASE_URL=""
   <!-- add login radious key then secret then site url and then app name  -->
  variables are - NEXT_PUBLIC_LOGINRADIUS_API_KEY=""

### Prerequisites

Ensure you have the following installed:

- [Node.js](https://nodejs.org/) (version v20.18.0 or higher)
- Package manager: `npm`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thoughtswinsystems/amplifi.git
   cd frontend
   ```
