export type ValidationState = {
  ok: boolean;
  message: string;
};

export type FormCharge = {
  id: string;
  label: string;
  units: number;
  rate: number;
};

export function validateClientForm(
  name: string,
  email: string,
  postalCode: string,
): ValidationState {
  const trimmedName = name.trim();
  const trimmedEmail = email.trim();
  const trimmedPostal = postalCode.trim();

  if (!trimmedName) {
    return { ok: false, message: "Name is required" };
  }
  if (trimmedName.length < 2) {
    return { ok: false, message: "Name must be at least 2 characters" };
  }
  if (!trimmedEmail) {
    return { ok: false, message: "Email is required" };
  }
  if (!trimmedEmail.includes("@") || !trimmedEmail.includes(".")) {
    return { ok: false, message: "Email format is invalid" };
  }
  if (!/^\d{5}$/.test(trimmedPostal)) {
    return { ok: false, message: "Postal code must be 5 digits" };
  }

  return { ok: true, message: "" };
}

export function buildChargeTotals(charges: FormCharge[], discountPct: number): {
  subtotal: number;
  discount: number;
  tax: number;
  total: number;
} {
  let subtotal = 0;
  for (const charge of charges) {
    subtotal += charge.units * charge.rate;
  }

  const normalizedSubtotal = Math.round(subtotal * 100) / 100;
  const appliedDiscount = Math.round(normalizedSubtotal * Math.max(0, discountPct) * 100) / 100;
  const discountedSubtotal = Math.max(0, normalizedSubtotal - appliedDiscount);
  const tax = Math.round(discountedSubtotal * 0.0825 * 100) / 100;
  const total = Math.round((discountedSubtotal + tax) * 100) / 100;

  return {
    subtotal: normalizedSubtotal,
    discount: appliedDiscount,
    tax,
    total,
  };
}
