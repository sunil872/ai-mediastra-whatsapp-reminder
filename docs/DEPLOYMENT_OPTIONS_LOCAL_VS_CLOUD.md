# Refill Reminder: Store Data and the Cloud

**For:** Mediastra management · **Date:** October 2026

---

## 1. The requirement

| Item | Agreed |
|---|---|
| Application | Mediastra is a **cloud web application**; Refill Reminder is used from inside it |
| Data | Sales data is on each store's own computer / local database |
| Store user | Uploads sales and sees results; never touches code |
| WhatsApp | Each store has its own WhatsApp Business account |

---

## 2. Short answer

**A cloud web application can't predict refills from data that stays only on the store computer.**

To work, Refill Reminder needs the store's sales data in the **same place the application runs**, which is the Mediastra cloud.

---

## 3. Why

1. **The work happens where the application runs.** Refill Reminder reads each patient's purchase history, predicts the next refill date and builds the reminder list. In a cloud web application, all of this happens on the cloud server.
2. **The cloud can't read the store computer.** The store PC sits behind the shop's internet router. The shop can reach the internet, but the cloud can't open files or databases on that PC.
3. **Opening store PCs to the internet isn't safe.** Making 20+ billing PCs reachable from outside would expose patient and billing data to attack. We don't recommend it.
4. **Uploading already means the cloud.** When the pharmacist uploads the daily sales file in the web application, the file goes to the cloud server. That is how every web application works.

---

## 4. Recommended approach: secure, isolated store data in the Mediastra cloud

| Protection | What it means |
|---|---|
| **India-region cloud** | Data is stored and processed in India (e.g. AWS Mumbai) |
| **Separate database per store** | Each store's data is kept apart; stores can never see each other's data |
| **Separate model per store** | Each store's predictions are trained only on its own sales |
| **Tenant / store login** | Mediastra's existing logins control who sees which store |
| **Encryption** | Data encrypted in storage and in transit (HTTPS) |
| **Per-store access keys** | Each store's Refill Reminder access is separately protected |
| **Automatic backups** | Daily backups with point-in-time restore; no data loss if a store PC fails |
| **Audit logs** | Every upload, approval and message send is recorded |
| **Data-processing agreement** | Written terms on how store data is used and protected (India DPDP Act) |

**Only the WhatsApp message itself** (patient name, medicine, due date) goes to WhatsApp, as for any WhatsApp reminder.

---

## 5. How it works for the store user

1. Log in to Mediastra and open **Refill Reminder**.
2. **Upload** the sales history once (3+ years), then yesterday's sales each day.
3. See today's reminder list, **approve**, then **send** from the store's own WhatsApp number.

Nothing to install and no code. It works from any computer with a browser.

---

## 6. Decision needed

Please confirm that each store's sales data may be kept in **Mediastra's India-region cloud**, in a **separate, encrypted database per store**, for Refill Reminder.

We then start with a **2–3 store pilot**, followed by a gradual roll-out to all stores.
