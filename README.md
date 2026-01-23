# RT-SEG

This project requires a configuration file for database authentication.

### 🔐 Configuration Setup

Please ensure the `sdb_login.json` file is located in the `/data/` directory with the following structure:

```json
{
  "user": "", 
  "pwd": "", 
  "ns": "NR",
  "db": "RT",
  "url": "wss://domain"
}
