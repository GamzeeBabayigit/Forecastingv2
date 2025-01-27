from django.db import models

# Create your models here.

class SalesData(models.Model):
    material = models.CharField(max_length=100)
    date = models.DateField()
    quantity = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.material} - {self.date} - {self.quantity}"

    class Meta:
        ordering = ['date']
